"""Algebraic multigrid (AMG) solver for the pressure-Poisson / Darcy systems.

This is a sibling of `DarcySolver`, deliberately interface-compatible with it
(`set_linear_system` -> `generate_preconditioner` -> `solve_pcg(X0)`), so it drops
into `StokesSolver.poisson_step` and into any NetworkManager / VolumeManager CSR
system with no other changes. `DarcySolver` is left untouched; both remain
selectable.

Why AMG (see `stokes_solver.md` s10.3): diagonal-preconditioned CG needs
O(N^{1/3})-ish iterations that blow up on real rock (a 250^3 Bentheimer Poisson
solve took ~2600 CG iterations). AMG makes the V-cycle count (nearly)
size-independent by coarsening the *matrix graph* directly -- which also makes it
agnostic to whether the CSR came from voxels (`VolumeManager`) or a pore network
(`NetworkManager`).

Two backends, selected by `backend=`:

- ``"native"`` (default, the real deliverable): our own aggregation + Numba
  V-cycle. Setup runs once in Python; the per-cycle hot loops are module-level
  ``@njit`` kernels, following the pattern in `darcySolver.py`.
- ``"pyamg"`` (permanent benchmark option): builds the hierarchy with pyamg. Kept
  as a first-class, reproducible comparison path -- an apples-to-apples reference
  for the native backend's iteration counts and solution. pyamg is imported
  lazily, so it is only required when this backend is used.

Sign convention: the assembled Laplacian is symmetric *negative*-definite
(diagonal < 0). CG/AMG want an SPD operator, so internally we solve the
equivalent system ``(-A) x = (-b)`` when the diagonal is negative; the solution
``x`` is unchanged. `_spd_sign` records the flip.
"""

import numpy as np
import scipy.sparse as sp

from pyflowsolver.solver import Solver


class MultigridSolver(Solver):
    BACKENDS = ("native", "pyamg")

    DEFAULT_PARAMS = {
        "max_iterations": 500,     # V-cycles / MG-PCG iters (MG needs very few)
        "target_error": 1e-07,     # relative-residual stopping tolerance
    }

    AGGREGATION = ("smoothed", "unsmoothed")

    def __init__(self, backend="native", aggregation="smoothed", **params):
        if backend not in self.BACKENDS:
            raise Exception(f"backend must be one of {self.BACKENDS}, got {backend!r}")
        if aggregation not in self.AGGREGATION:
            raise Exception(f"aggregation must be one of {self.AGGREGATION}, "
                            f"got {aggregation!r}")
        self.backend = backend
        # Prolongator style for the native backend: "smoothed" (Jacobi-smoothed
        # aggregation, size-independent convergence) or "unsmoothed"
        # (piecewise-constant, simpler but weaker). Ignored by the pyamg backend.
        self.aggregation = aggregation
        self.params = self.DEFAULT_PARAMS.copy()

        # Linear system (project CSR convention: row_ptr has N entries).
        self.a_sparse_array = None
        self.b_array = None
        self.N = None
        self.V = None

        # scipy SPD view of A (sign-corrected) built once in set_linear_system.
        self._A_spd = None
        self._spd_sign = 1.0

        # Hierarchy handle (pyamg MultilevelSolver, or native level list).
        self.hierarchy = None

        # Diagnostics from the last solve.
        self.x = None
        self.error = None
        self.iteration = None

        for key, value in params.items():
            if key in self.DEFAULT_PARAMS:
                self.params[key] = value
            else:
                raise Exception(f"{key} is not a valid parameter,"
                                f"parameters are {list(self.params.keys())}")

    # ------------------------------------------------------------------ #
    # System setup (interface-compatible with DarcySolver)
    # ------------------------------------------------------------------ #
    def set_linear_system(self, a_sparse_array, b_array):
        """Store A, b and build the sign-corrected SPD scipy view of A.

        Accepts the same CSR the rest of the project uses (``row_ptr`` length N),
        and also the scipy convention (length N+1, last entry == nnz) for parity
        with `DarcySolver.set_linear_system`.
        """
        val = a_sparse_array["val"]
        col_idx = a_sparse_array["col_idx"]
        row_ptr = a_sparse_array["row_ptr"]

        if val.size != col_idx.size:
            raise Exception(f"Inconsistent non-zero values for 'val' ({val.size}) "
                            f"and 'col_idx' ({col_idx.size})")

        if row_ptr.size == b_array.size:
            indptr = np.empty(row_ptr.size + 1, dtype=np.int64)
            indptr[:-1] = row_ptr
            indptr[-1] = val.size
        elif row_ptr.size == b_array.size + 1:
            if row_ptr[-1] != val.size:
                raise Exception("Scipy format row_ptr last value must equal nnz, "
                                f"was {row_ptr[-1]}")
            indptr = row_ptr.astype(np.int64)
        else:
            raise Exception("b_array and row_ptr lengths did not match")

        self.a_sparse_array = a_sparse_array
        self.b_array = b_array
        self.N = b_array.size
        self.V = val.size

        A = sp.csr_matrix((val.astype(np.float64), col_idx.astype(np.int64), indptr),
                          shape=(self.N, self.N))
        # Symmetric negative-definite Laplacian -> flip to SPD for CG/AMG.
        self._spd_sign = -1.0 if A.diagonal().sum() < 0 else 1.0
        self._A_spd = (self._spd_sign * A).tocsr()

    def generate_preconditioner(self, preconditioner=None):
        """Build the AMG hierarchy (once); the geometry/matrix is fixed.

        The ``preconditioner`` argument exists only for signature parity with
        `DarcySolver.generate_preconditioner`; the hierarchy *is* the
        preconditioner here, so the value is ignored.
        """
        if self._A_spd is None:
            raise Exception("call set_linear_system before generate_preconditioner")
        if self.backend == "pyamg":
            self._setup_pyamg()
        else:
            self._setup_native()

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    def solve_pcg(self, X0=None):
        """MG-preconditioned CG. Returns (x, relative_error, iterations)."""
        return self._solve(X0=X0, accel="cg")

    def solve_mg(self, X0=None):
        """Standalone V-cycle iteration. Returns (x, relative_error, iterations)."""
        return self._solve(X0=X0, accel=None)

    def _solve(self, X0, accel):
        if self.hierarchy is None:
            self.generate_preconditioner()
        b_spd = self._spd_sign * self.b_array
        if self.backend == "pyamg":
            x, err, it = self._solve_pyamg(b_spd, X0, accel)
        else:
            x, err, it = self._solve_native(b_spd, X0, accel)
        self.x, self.error, self.iteration = x, err, it
        return x, err, it

    # ------------------------------------------------------------------ #
    # pyamg backend (permanent benchmark reference)
    # ------------------------------------------------------------------ #
    def _setup_pyamg(self):
        import pyamg  # lazy: only needed for this backend
        # Smoothed aggregation on the SPD operator; hierarchy reused every solve.
        self.hierarchy = pyamg.smoothed_aggregation_solver(self._A_spd)

    def _solve_pyamg(self, b_spd, X0, accel):
        residuals = []
        x = self.hierarchy.solve(
            b_spd,
            x0=X0,
            tol=self.params["target_error"],
            maxiter=self.params["max_iterations"],
            accel=accel,
            residuals=residuals,
        )
        # residuals[0] is the initial residual norm; iters = number of cycles.
        r0 = residuals[0] if residuals else 0.0
        error = (residuals[-1] / r0) if r0 > 0 else 0.0
        iterations = max(len(residuals) - 1, 0)
        return x, error, iterations

    # ------------------------------------------------------------------ #
    # native backend: unsmoothed aggregation AMG + Numba V-cycle
    # ------------------------------------------------------------------ #
    # Setup knobs (aggregation coarsening + smoother). Deliberately plain.
    MAX_COARSE = 50        # stop coarsening at/below this many unknowns
    MAX_LEVELS = 25        # hard cap on hierarchy depth
    JACOBI_OMEGA = 2.0 / 3.0   # weighted-Jacobi relaxation factor
    PRE_SWEEPS = 2
    POST_SWEEPS = 2

    def _setup_native(self):
        """Build the AMG hierarchy once (Python; the geometry/matrix is fixed).

        Unsmoothed aggregation: greedy standard aggregation on the matrix graph,
        piecewise-constant (column-normalized) tentative prolongator P, R = P^T,
        Galerkin coarse operator A_c = R A P (scipy triple product, setup-only).
        Each level is stored in the project CSR convention with preallocated
        V-cycle scratch; the coarsest level keeps a dense inverse for the direct
        solve. Solve-time work touches only numpy + the @njit cycle kernels.
        """
        self.levels = []
        A = self._A_spd.tocsr()

        for _ in range(self.MAX_LEVELS):
            level = self._make_level(A)
            self.levels.append(level)
            n = A.shape[0]
            if n <= self.MAX_COARSE:
                level["coarse_inv"] = _safe_inverse(A.toarray())
                break

            agg, n_coarse = _standard_aggregation(
                A.indptr.astype(np.int64), A.indices.astype(np.int64)
            )
            if n_coarse >= n or n_coarse == 0:
                # Coarsening stalled -> make this the coarse (direct) level.
                level["coarse_inv"] = _safe_inverse(A.toarray())
                break

            P = _tentative_prolongator(agg, n_coarse)   # (n x n_coarse) scipy csr
            if self.aggregation == "smoothed":
                P = _smooth_prolongator(A, P)           # one Jacobi step on P
            R = P.T.tocsr()
            level["P"] = _to_project_csr(P)
            level["R"] = _to_project_csr(R)
            A = (R @ A @ P).tocsr()                      # Galerkin coarse operator

        self.hierarchy = self.levels

    def _make_level(self, A):
        """Package a scipy csr A as a project-CSR level with V-cycle scratch."""
        n = A.shape[0]
        csr = _to_project_csr(A)
        diag = A.diagonal().astype(np.float64)
        diag_inv = np.where(diag != 0.0, 1.0 / diag, 0.0)
        return {
            "n": n,
            "val": csr["val"], "col_idx": csr["col_idx"], "row_ptr": csr["row_ptr"],
            "diag_inv": diag_inv,
            "x": np.zeros(n, dtype=np.float64),
            "b": np.zeros(n, dtype=np.float64),
            "r": np.zeros(n, dtype=np.float64),
            "tmp": np.zeros(n, dtype=np.float64),
        }

    def _vcycle(self, l):
        """Recursive V-cycle: relax, restrict residual, correct, relax."""
        lvl = self.levels[l]
        if "coarse_inv" in lvl:
            lvl["x"][:] = lvl["coarse_inv"] @ lvl["b"]
            return

        _weighted_jacobi(lvl["val"], lvl["col_idx"], lvl["row_ptr"],
                         lvl["b"], lvl["x"], lvl["diag_inv"],
                         self.JACOBI_OMEGA, self.PRE_SWEEPS, lvl["tmp"])

        # r = b - A x
        _csr_residual(lvl["val"], lvl["col_idx"], lvl["row_ptr"],
                      lvl["b"], lvl["x"], lvl["r"])

        # Restrict residual to the coarse RHS; start the coarse guess at 0.
        coarse = self.levels[l + 1]
        R = lvl["R"]
        _csr_matvec(R["val"], R["col_idx"], R["row_ptr"], lvl["r"], coarse["b"])
        coarse["x"][:] = 0.0

        self._vcycle(l + 1)

        # Prolongate the coarse correction and add it: x += P e_c
        P = lvl["P"]
        _csr_matvec_add(P["val"], P["col_idx"], P["row_ptr"], coarse["x"], lvl["x"])

        _weighted_jacobi(lvl["val"], lvl["col_idx"], lvl["row_ptr"],
                         lvl["b"], lvl["x"], lvl["diag_inv"],
                         self.JACOBI_OMEGA, self.POST_SWEEPS, lvl["tmp"])

    def _apply_preconditioner(self, r):
        """One V-cycle with rhs r and a zero initial guess -> M^{-1} r."""
        top = self.levels[0]
        top["b"][:] = r
        top["x"][:] = 0.0
        self._vcycle(0)
        return top["x"].copy()

    def _solve_native(self, b_spd, X0, accel):
        top = self.levels[0]
        val, col, ptr = top["val"], top["col_idx"], top["row_ptr"]
        b = np.ascontiguousarray(b_spd, dtype=np.float64)
        x = np.zeros_like(b) if X0 is None else X0.astype(np.float64).copy()
        max_it = self.params["max_iterations"]
        tol = self.params["target_error"]

        b_norm = np.linalg.norm(b)
        if b_norm == 0.0:
            return np.zeros_like(b), 0.0, 0

        if accel is None:
            return self._iterate_vcycles(val, col, ptr, b, x, b_norm, max_it, tol)
        return self._iterate_mgpcg(val, col, ptr, b, x, b_norm, max_it, tol)

    def _iterate_vcycles(self, val, col, ptr, b, x, b_norm, max_it, tol):
        """Standalone V-cycle iteration to the residual tolerance."""
        r = np.empty_like(b)
        error = np.inf
        it = 0
        for it in range(1, max_it + 1):
            _csr_residual(val, col, ptr, b, x, r)
            error = np.linalg.norm(r) / b_norm
            if error <= tol:
                return x, error, it - 1
            x = x + self._apply_preconditioner(r)   # x <- x + M^{-1} r
        _csr_residual(val, col, ptr, b, x, r)
        return x, np.linalg.norm(r) / b_norm, it

    def _iterate_mgpcg(self, val, col, ptr, b, x, b_norm, max_it, tol):
        """MG-preconditioned CG: the preconditioner apply is one V-cycle."""
        r = np.empty_like(b)
        _csr_residual(val, col, ptr, b, x, r)      # r = b - A x
        z = self._apply_preconditioner(r)
        p = z.copy()
        rz = float(r @ z)
        Ap = np.empty_like(b)
        error = np.linalg.norm(r) / b_norm
        it = 0
        for it in range(1, max_it + 1):
            if error <= tol:
                return x, error, it - 1
            _csr_matvec(val, col, ptr, p, Ap)
            pAp = float(p @ Ap)
            if pAp == 0.0:
                break
            alpha = rz / pAp
            x += alpha * p
            r -= alpha * Ap
            error = np.linalg.norm(r) / b_norm
            if error <= tol:
                return x, error, it
            z = self._apply_preconditioner(r)
            rz_new = float(r @ z)
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new
        return x, error, it


# ====================================================================== #
# Setup helpers (Python / scipy, run once)
# ====================================================================== #
def _to_project_csr(M):
    """scipy csr -> project CSR dict (row_ptr length N, not N+1)."""
    M = M.tocsr()
    M.sort_indices()
    return {
        "val": M.data.astype(np.float64),
        "col_idx": M.indices.astype(np.int64),
        "row_ptr": M.indptr[:-1].astype(np.int64).copy(),
    }


def _tentative_prolongator(agg, n_coarse):
    """Piecewise-constant, column-normalized prolongator P (n x n_coarse).

    Column j (aggregate j) holds the normalized constant near-nullspace vector
    on its member nodes: entry 1/sqrt(|aggregate j|). This keeps R A P scaled.
    """
    n = agg.size
    counts = np.bincount(agg, minlength=n_coarse).astype(np.float64)
    data = 1.0 / np.sqrt(counts[agg])
    rows = np.arange(n, dtype=np.int64)
    return sp.csr_matrix((data, (rows, agg)), shape=(n, n_coarse))


def _smooth_prolongator(A, P_tent):
    """Smoothed-aggregation prolongator: one weighted-Jacobi step on P_tent.

        P = (I - omega * D^-1 A) P_tent,   omega = 4 / (3 * rho(D^-1 A))

    Smoothing the piecewise-constant interpolation is what makes aggregation
    multigrid converge at a size-independent rate (Vanek, Mandel & Brezina 1996);
    `rho` is the spectral radius of the Jacobi iteration matrix, estimated by a
    few power iterations at setup (once). All scipy, setup-only.
    """
    A = A.tocsr()
    d = A.diagonal()
    d_inv = np.where(d != 0.0, 1.0 / d, 0.0)
    Dinv_A = sp.diags(d_inv) @ A
    rho = _estimate_spectral_radius(Dinv_A)
    omega = (4.0 / 3.0) / rho if rho > 0.0 else 0.0
    return (P_tent - omega * (Dinv_A @ P_tent)).tocsr()


def _estimate_spectral_radius(M, iters=15, seed_size=None):
    """Power-iteration estimate of the spectral radius of sparse M."""
    n = M.shape[0]
    v = np.ones(n, dtype=np.float64)
    v /= np.linalg.norm(v)
    rho = 0.0
    for _ in range(iters):
        w = M @ v
        nw = np.linalg.norm(w)
        if nw == 0.0:
            return 0.0
        rho = nw
        v = w / nw
    return rho


def _safe_inverse(dense):
    """Dense inverse for the coarse direct solve; pinv if (near-)singular."""
    try:
        return np.linalg.inv(dense)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(dense)


# ====================================================================== #
# Aggregation (@njit; integer graph passes over the matrix CSR)
# ====================================================================== #
try:
    from numba import njit, prange
except Exception:  # pragma: no cover - numba is a hard dep, but stay importable
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if (a and callable(a[0])) is False else a[0]
    prange = range


@njit(cache=True)
def _standard_aggregation(indptr, indices):
    """Greedy standard (Vanek) aggregation on the symmetric matrix graph.

    `indptr`/`indices` are scipy-CSR (indptr length N+1) and include the
    diagonal, which is skipped. Returns (agg, n_coarse) where agg[i] is the
    aggregate id of node i. Three passes: seed aggregates from nodes whose whole
    neighborhood is free, attach leftovers to an existing (pass-1) aggregate,
    then make any remaining nodes singletons.
    """
    n = indptr.size - 1
    agg = -np.ones(n, dtype=np.int64)
    next_agg = 0

    # Pass 1: seed aggregates around fully-unaggregated neighborhoods.
    for i in range(n):
        if agg[i] != -1:
            continue
        has_aggregated_nbr = False
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if j != i and agg[j] != -1:
                has_aggregated_nbr = True
                break
        if has_aggregated_nbr:
            continue
        agg[i] = next_agg
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if j != i:
                agg[j] = next_agg
        next_agg += 1

    # Snapshot so pass 2 only attaches to pass-1 aggregates (no chaining).
    agg1 = agg.copy()

    # Pass 2: attach leftovers to a neighboring pass-1 aggregate.
    for i in range(n):
        if agg[i] != -1:
            continue
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if j != i and agg1[j] != -1:
                agg[i] = agg1[j]
                break

    # Pass 3: anything still free becomes its own aggregate.
    for i in range(n):
        if agg[i] == -1:
            agg[i] = next_agg
            next_agg += 1

    return agg, next_agg


# ====================================================================== #
# V-cycle hot loops (@njit; project CSR: row_ptr length N)
# ====================================================================== #
@njit(parallel=True, cache=True)
def _csr_matvec(val, col_idx, row_ptr, x, out):
    """out = A x  (project CSR; out preallocated, size = number of rows)."""
    n = row_ptr.size
    nnz = val.size
    for row in prange(n):
        start = row_ptr[row]
        stop = row_ptr[row + 1] if (row + 1) < n else nnz
        acc = 0.0
        for k in range(start, stop):
            acc += val[k] * x[col_idx[k]]
        out[row] = acc


@njit(parallel=True, cache=True)
def _csr_matvec_add(val, col_idx, row_ptr, x, out):
    """out += A x  (project CSR; used for the prolongation correction)."""
    n = row_ptr.size
    nnz = val.size
    for row in prange(n):
        start = row_ptr[row]
        stop = row_ptr[row + 1] if (row + 1) < n else nnz
        acc = 0.0
        for k in range(start, stop):
            acc += val[k] * x[col_idx[k]]
        out[row] += acc


@njit(parallel=True, cache=True)
def _csr_residual(val, col_idx, row_ptr, b, x, out):
    """out = b - A x  (project CSR)."""
    n = row_ptr.size
    nnz = val.size
    for row in prange(n):
        start = row_ptr[row]
        stop = row_ptr[row + 1] if (row + 1) < n else nnz
        acc = 0.0
        for k in range(start, stop):
            acc += val[k] * x[col_idx[k]]
        out[row] = b[row] - acc


@njit(parallel=True, cache=True)
def _weighted_jacobi(val, col_idx, row_ptr, b, x, diag_inv, omega, sweeps, tmp):
    """`sweeps` weighted-Jacobi relaxations in place: x += omega D^-1 (b - A x).

    `tmp` is preallocated scratch (size N). Reads the whole current x each sweep
    (Jacobi, not Gauss-Seidel), so it is safe under prange.
    """
    n = row_ptr.size
    nnz = val.size
    for _ in range(sweeps):
        for row in prange(n):
            start = row_ptr[row]
            stop = row_ptr[row + 1] if (row + 1) < n else nnz
            acc = 0.0
            for k in range(start, stop):
                acc += val[k] * x[col_idx[k]]
            tmp[row] = x[row] + omega * diag_inv[row] * (b[row] - acc)
        for row in prange(n):
            x[row] = tmp[row]
