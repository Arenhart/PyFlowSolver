import multiprocessing
import psutil

import numpy as np
from numba import njit, prange

from pyflowsolver.solver import Solver

class DarcySolver(Solver):
    PRECONDITIONERS = [
        "inverse_diagonal",
    ]
    DEFAULT_PARAMS = {
        "max_iterations" : 5000,
        "target_error" : 1e-07,
    }

    def __init__(self, **params):
        self.params = self.DEFAULT_PARAMS.copy()
        self.a_sparse_array = None
        self.b_array = None
        self.preconditioner = None
        self.scipy_format = None
        self.V = None
        self.N = None
        self.x = None
        self.error = None
        self.iteration = None

        for key, value in params.items():
            if key in self.DEFAULT_PARAMS.keys():
                self.default_params[key] = value
            else:
                raise Exception(f"{key} is not a valid parameter,"
                                f"parameters are {list(self.default_params.keys())}")
        
        
    def set_linear_system(self, a_sparse_array, b_array):
        """
        for a N x N array with V non-zero values
        Array formats:
            a_sparse:
                'val': V length
                'col_idx': V length
                'row_ptr': N length, or N + 1 if scipy format
            b_dense: N length
        """
        if a_sparse_array["val"].size != a_sparse_array["col_idx"].size:
            raise Exception(f"Inconsistent non-zero values for 'val'"
                            f"({a_sparse_array['val'].size}) and 'col_idx'"
                            f"({a_sparse_array['col_idx'].size})")
        if a_sparse_array["row_ptr"].size == b_array.size:
            scipy_format = False
        elif a_sparse_array["row_ptr"].size == (b_array.size + 1):
            if a_sparse_array["row_ptr"][-1] != a_sparse_array["val"].size:
                raise Exception(f"Scipy format row_ptr last value must be equal to V"
                                f", was {a_sparse_array['row_ptr'][-1]}")
            scipy_format = True
        else:
            raise Exception(f"b_array and row_ptr lengths did not match")
        self.scipy_format = scipy_format
        self.a_sparse_array = a_sparse_array
        self.b_array = b_array
        self.N = b_array.size
        self.V = a_sparse_array["val"].size


    def generate_preconditioner(self, preconditioner):

        if preconditioner == "inverse_diagonal":
            P_val, P_col_idx, P_row_ptr = self._get_diagonal_preconditioner(
                A_val=self.a_sparse_array["val"],
                A_col_idx=self.a_sparse_array["col_idx"],
                A_row_ptr=self.a_sparse_array["row_ptr"],
            )
        self.preconditioner = {
            "val": P_val,
            "col_idx": P_col_idx,
            "row_ptr": P_row_ptr,
        }


    def solve_cg(self):

        self.x, self.error, self.iteration = self._solve_cg(
            A_val=self.a_sparse_array["val"],
            A_col_idx=self.a_sparse_array["col_idx"],
            A_row_ptr=self.a_sparse_array["row_ptr"], 
            b=self.b_array,
            max_iterations=self.params["max_iterations"], # sqrt(V)
            target_error=self.params["target_error"], # 1.0e-6
            X0=np.zeros_like(self.b_array),
        )
        return self.x, self.error, self.iteration

    def solve_pcg(self):

        self.x, self.error, self.iteration = self._solve_pcg(
            A_val=self.a_sparse_array["val"],
            A_col_idx=self.a_sparse_array["col_idx"],
            A_row_ptr=self.a_sparse_array["row_ptr"],
            P_val=self.preconditioner["val"],
            P_col_idx=self.preconditioner["col_idx"],
            P_row_ptr=self.preconditioner["row_ptr"],
            b=self.b_array,
            max_iterations=self.params["max_iterations"], # sqrt(V)
            target_error=self.params["target_error"], # 1.0e-6
            X0=np.zeros_like(self.b_array),
            threads=4
        )
        return self.x, self.error, self.iteration


    @staticmethod
    @njit(parallel=True)
    def _get_diagonal_preconditioner(
        A_val, 
        A_col_idx, 
        A_row_ptr, 
    ): # f(v, A): v = A*v
        diagonal_n = A_row_ptr.size
        P_val = np.empty(diagonal_n, dtype=np.float64)
        P_col_idx = np.arange(diagonal_n, dtype=np.int32)
        P_row_ptr = np.arange(diagonal_n, dtype=np.int32)
        for row in range(diagonal_n):
            start = A_row_ptr[row]
            if row < (A_row_ptr.size - 1):
                stop = A_row_ptr[row + 1]
            else:
                stop = A_val.size

            for linear_index in range(start, stop):
                column = A_col_idx[linear_index]
                if column == row:
                    v = A_val[linear_index]
                    P_val[row] = 1/v
        return P_val, P_col_idx, P_row_ptr


    @staticmethod
    @njit
    def _solve_cg(
        A_val,
        A_col_idx,
        A_row_ptr, 
        b,
        max_iterations, # sqrt(V)
        target_error, # 1.0e-6
        X0,
    ):
        #Reference: https://repository.lsu.edu/cgi/viewcontent.cgi?article=1254&context=honors_etd

        x = X0.copy()
        r = b.copy()
        m = np.empty(1, dtype=np.float64)
        m[0] = _square_sum_vector(r) # f(x:vector) = x'*x
        m_last = np.empty(1, dtype=np.float64)
        p = r.copy()
        alpha = np.empty(1, dtype=np.float64)
        beta = np.empty(1, dtype=np.float64)
        iteration = 0
        for _ in range(max_iterations):
            iteration += 1
            alpha[0] = m[0] / _scalar_product(
                p, 
                A_val, 
                A_col_idx, 
                A_row_ptr,
                ) # scalar_product = p'*A*p
            _add_product(x, alpha[0], p) # f(x: vector, y: scalar, z:vector): x += y * z
            _recalc_residuals_jit(r, A_val, A_col_idx, A_row_ptr, b, x)
            m_last[0] = m[0]
            m[0] = _square_sum_vector(r)
            beta[0] = m[0] / m_last[0]
            _multiply_and_add(
                p, 
                r, 
                beta[0],
            ) # f(x:vector, y:vector, z:scalar): x = y + z * x
            error = np.sqrt(_square_sum_vector(r) 
                            / _square_sum_vector(b)
            )
            if error <= target_error:
                return x, error, iteration

        return x, error, iteration
    
    @staticmethod
    @njit
    def _solve_pcg(
        A_val,
        A_col_idx,
        A_row_ptr,
        P_val,
        P_col_idx,
        P_row_ptr,
        b,
        max_iterations,  # sqrt(n) for n x n system
        target_error,  # 1.0e-6
        X0,
        threads,
    ):
        # Reference: https://repository.lsu.edu/cgi/viewcontent.cgi?article=1254&context=honors_etd

        x = X0.copy()
        r = b.copy()
        m = np.empty(1, dtype=np.float64)
        m[0] = _scalar_product(
            r,
            P_val,
            P_col_idx,
            P_row_ptr,
            threads,
        )  # scalar_product = p'*A*p
        m_last = np.empty(1, dtype=np.float64)
        p = r.copy()
        p[:] = _vector_array_multiply(
            p,
            P_val,
            P_col_idx,
            P_row_ptr,
            threads,
        )  # f(v, A): v = A*v
        alpha = np.empty(1, dtype=np.float64)
        beta = np.empty(1, dtype=np.float64)
        iteration = 0
        for _ in range(max_iterations):
            iteration += 1
            alpha[0] = m[0] / _scalar_product(
                p,
                A_val,
                A_col_idx,
                A_row_ptr,
                threads,
            )  # scalar_product = p'*A*p
            _add_product(x, alpha[0], p, threads)  # f(x: vector, y: scalar, z:vector): x += y * z
            _recalc_residuals_jit(r, A_val, A_col_idx, A_row_ptr, b, x)
            m_last[0] = m[0]
            m[0] = _scalar_product(
                r,
                P_val,
                P_col_idx,
                P_row_ptr,
                threads,
            )  # scalar_product = p'*A*p
            beta[0] = m[0] / m_last[0]
            p[:] = _multiply_array_and_add(
                p,
                r,
                beta[0],
                P_val,
                P_col_idx,
                P_row_ptr,
                threads,
            )  # f(x:vector, y:vector, z:scalar, A:array): x = A * y + z * x
            error = np.sqrt(_square_sum_vector(r, threads) / _square_sum_vector(b, threads))
            if error <= target_error:
                return x, error, iteration

        return x, error, iteration


@njit(parallel=True)
def _square_sum_vector(v, threads):
    # f(v:vector) = v'*v
    partial_sum = np.zeros(threads, dtype=np.float64)
    n = v.size

    for w in prange(threads):
        thread_start = w * n // threads
        thread_end = (w + 1) * n // threads
        for i in range(thread_start, thread_end):
            partial_sum[w] += v[i] ** 2
    return partial_sum.sum()


@njit(parallel=True)
def _scalar_product(
    v,
    A_val,
    A_col_idx,
    A_row_ptr,
    threads,
):
    # f(v: vector[n], A:array[n, n]) = x'*A*x
    partial_sum = np.zeros(threads, dtype=np.float64)
    n = v.size

    for w in prange(threads):
        thread_start = w * n // threads
        thread_end = (w + 1) * n // threads
        for i in range(thread_start, thread_end):
            A_start = A_row_ptr[i]
            if (i + 1) < n:
                A_stop = A_row_ptr[i + 1]
            else:
                A_stop = A_val.size
            for A_linear_index in range(A_start, A_stop):
                j = A_col_idx[A_linear_index]
                partial_sum[w] += v[i] * A_val[A_linear_index] * v[j]
    return partial_sum.sum()


@njit(parallel=True)
def _add_product(v, x, u, threads):
    # f(v: vector, x: scalar, u:vector): v += x * u
    n = v.size

    for w in prange(threads):
        thread_start = w * n // threads
        thread_end = (w + 1) * n // threads
        for i in range(thread_start, thread_end):
            v[i] += x * u[i]


@njit(parallel=True)
def _recalc_residuals_jit(r, val, col_idx, row_ptr, condensed_b, X):
    # residuals = np.zeros(condensed_b.size, dtype=np.float64)
    r[:] = condensed_b

    for row in range(row_ptr.size):
        start = row_ptr[row]
        if row < (row_ptr.size - 1):
            stop = row_ptr[row + 1]
        else:
            stop = val.size

        for index in range(start, stop):
            v = val[index]
            column = col_idx[index]
            r[row] -= v * X[column]


@njit(parallel=True)
def _subtract_product_of_product(
    v,
    x,
    A_val,
    A_col_idx,
    A_row_ptr,
    u,
    threads,
):
    # f(v:vector, x:scalar, A:array, u:vector): v -= x * A * u
    n = v.size

    for w in prange(threads):
        thread_start = w * n // threads
        thread_end = (w + 1) * n // threads
        for i in range(thread_start, thread_end):
            A_start = A_row_ptr[i]
            if (i + 1) < n:
                A_stop = A_row_ptr[i + 1]
            else:
                A_stop = A_val.size
            for A_linear_index in range(A_start, A_stop):
                j = A_col_idx[A_linear_index]
                v[i] -= x * A_val[A_linear_index] * u[j]


@njit(parallel=True)
def _multiply_and_add(
    v,
    u,
    x,
    threads,
):
    # f(v:vector, u:vector, x:scalar): v = u + x * v
    n = v.size

    for w in prange(threads):
        thread_start = w * n // threads
        thread_end = (w + 1) * n // threads
        for i in range(thread_start, thread_end):
            v[i] = u[i] + x * v[i]


@njit
def _get_diagonal_preconditioner(
    A_val,
    A_col_idx,
    A_row_ptr,
    threads,
):  # f(v, A): v = A*v
    diagonal_n = A_row_ptr.size
    P_val = np.empty(diagonal_n, dtype=np.float64)
    P_col_idx = np.arange(diagonal_n, dtype=np.int32)
    P_row_ptr = np.arange(diagonal_n, dtype=np.int32)
    for row in range(diagonal_n):
        start = A_row_ptr[row]
        if row < (A_row_ptr.size - 1):
            stop = A_row_ptr[row + 1]
        else:
            stop = A_val.size

        for linear_index in range(start, stop):
            column = A_col_idx[linear_index]
            if column == row:
                v = A_val[linear_index]
                P_val[row] = 1 / v
    return P_val, P_col_idx, P_row_ptr





@njit  #
def _vector_array_multiply(
    p,
    val,
    col_idx,
    row_ptr,
    threads,
):  # f(v, A): v = A*v
    new_p = np.zeros_like(p)
    for row in range(row_ptr.size):
        start = row_ptr[row]
        if row < (row_ptr.size - 1):
            stop = row_ptr[row + 1]
        else:
            stop = val.size

        for index in range(start, stop):
            v = val[index]
            column = col_idx[index]
            new_p[row] += v * p[column]
    return new_p


@njit(parallel=True)
def _multiply_array_and_add(
    u,
    v,
    x,
    val,
    col_idx,
    row_ptr,
    threads,
):  # f(u:vector, v:vector, x:scalar, A:array): x = A * v + x * u
    new_p = _vector_array_multiply(
        v,
        val,
        col_idx,
        row_ptr,
        threads,
    )
    new_p += x * u
    return new_p
