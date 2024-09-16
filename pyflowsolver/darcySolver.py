import multiprocessing
import psutil

import numpy as np
from numba import njit, prange

from pyflowsolver.solver import Solver

class DarcySolver(Solver):
    def __init__(self):
        self.default_params = {
            "max_step" : 1/4,
            "step_adjustment" : 1/4,
            "initial_step" : 1/8,
            "max_iterations" : 5000,
            "target_error" : 1e-07,
        }


    def solve(self, A, b, X0=None, **params):

        def param(key):
            return params.get(key, self.default_params[key])

        if X0 is None:
            X0 = np.linspace(1, 0, num=b.size, dtype=np.float32)
                
        next_x = X0.copy()
        error = np.inf
        step = param("initial_step")
        residuals = self.calc_residuals(A, b, X0)
        diag = np.zeros(A.shape[0], dtype = np.float32)
        for i in range(A.shape[0]):
            diag[i] = A[i, i]
        iteration = 0

        for _ in range(param("max_iterations")):
            iteration += 1
            next_x = X0 - residuals * step / diag
            next_residuals = self.calc_residuals(A, b, next_x)
            next_error = (next_residuals**2).sum()/X0.size
            if next_error <= param("target_error"):
                return next_x, next_error, iteration
            elif (next_error < error):# and (next_x >= 0).all():
                residuals = next_residuals
                X0 = next_x
                error = next_error
                step += (param("max_step") - step) * param("step_adjustment")
            else:
                step = 1/step
                step += (step + 1/param("max_step"))
                step = 1/step
        else:
            return X0, error, iteration


    def calc_residuals(self, sparse_array, condensed_b, X):

        residuals = np.zeros(condensed_b.size, dtype=np.float32)
        for row, start, stop in sparse_array.row_iterator():
            residual = np.float32(0)
            for index in range(start, stop):
                val = sparse_array[index]
                column = sparse_array.col_idx[index]
                x_val = X[column]
                residual += val * x_val
            residual -= condensed_b[row]
            residuals[row] = residual
        return residuals

    def solve_jit(self, A, b, parallel=False, **params):

        max_step = params.get("max_step", self.default_params["max_step"])
        step_adjustment = params.get("step_adjustment", self.default_params["step_adjustment"])
        initial_step = params.get("initial_step", self.default_params["initial_step"])
        max_iterations = params.get("max_iterations", self.default_params["max_iterations"])
        target_error = params.get("target_error", self.default_params["target_error"])

        X0 = np.linspace(1, 0, num=b.size, dtype=np.float32)

        if not parallel:
            X0, error, iterations = self._solve_jit(
                A.val,
                A.col_idx,
                A.row_ptr,
                b,
                max_step,
                step_adjustment,
                initial_step,
                max_iterations,
                target_error,
                X0, 
                )
        else:

            if type(parallel) == int:
                threads = parallel
            elif parallel is True:
                threads = max(psutil.cpu_count(logical=False) - 2, 1)
            else:
                raise TypeError("Invalid type of parallel parameter, must be True, False or int")

            X0, error, iterations = self._solve_jit_parallel(
                A.val,
                A.col_idx,
                A.row_ptr,
                b,
                max_step,
                step_adjustment,
                initial_step,
                max_iterations,
                target_error,
                X0,
                threads=threads,
                )

        return X0, error, iterations
        

    @staticmethod
    @njit
    def _solve_jit(
        A_val,
        A_col_idx,
        A_row_ptr, 
        b,
        max_step,
        step_adjustment,
        initial_step,
        max_iterations,
        target_error,
        X0,
    ):
        
        A_rows = A_row_ptr.size
                
        next_x = X0.copy().astype(np.float32)
        error = np.inf
        step = np.float32(initial_step)
        residuals = _calc_residuals_jit(A_val, A_col_idx, A_row_ptr, b, X0).astype(np.float32)
        diag = np.zeros(A_rows, dtype = np.float32)
        iterations = 0
        for i in range(A_rows):
            diag[i] = _getitem(A_val, A_col_idx, A_row_ptr, i, i)
        for _ in range(max_iterations):
            iterations += 1
            next_x[:] = residuals[:]
            next_x *= step
            next_x /= diag
            next_x *= np.float32(-1)
            next_x += X0
            next_residuals = _calc_residuals_jit(A_val, A_col_idx, A_row_ptr, b, next_x)
            next_error = (next_residuals**2).sum()/X0.size
            if next_error <= target_error:
                return next_x, next_error, iterations
            elif (next_error < error) and (next_x >= 0).all():
                residuals = next_residuals
                X0[:] = next_x[:]
                error = next_error
                step += (max_step - step) * step_adjustment
            else:
                step = 1/step
                step += (step + 1/max_step)
                step = 1/step
            if step <= 1e-10:
                return X0, error, iterations
        else:
            return X0, error, iterations
        

    @staticmethod
    @njit
    def _solve_jit_parallel(
        A_val,
        A_col_idx,
        A_row_ptr, 
        b,
        max_step,
        step_adjustment,
        initial_step,
        max_iterations,
        target_error,
        X0,
        threads,
    ):
        
        A_rows = A_row_ptr.size
                
        next_x = X0.copy().astype(np.float32)
        error = np.array((np.inf,), dtype=np.float32)
        next_error = np.array((np.inf,), dtype=np.float32)
        step = np.float32(initial_step)
        residuals = _calc_residuals_jit_parallel(A_val, A_col_idx, A_row_ptr, b, X0, threads).astype(np.float32)
        next_residuals = np.empty_like(residuals)
        diag = np.zeros(A_rows, dtype = np.float32)
        iterations = 0
        for i in range(A_rows):
            diag[i] = _getitem(A_val, A_col_idx, A_row_ptr, i, i)
        for _ in range(max_iterations):
            iterations += 1
            _calc_next_x(next_x, residuals, step, diag, X0, threads)
            _update_resdiuals_and_error_parallel(
                next_residuals, 
                next_error, 
                A_val, 
                A_col_idx, 
                A_row_ptr, 
                b, 
                next_x, 
                threads,
            )
            #next_residuals = _calc_residuals_jit_parallel(A_val, A_col_idx, A_row_ptr, b, next_x, threads)
            #next_error = (next_residuals**2).sum()/X0.size
            if next_error[0] <= target_error:
                return next_x, next_error, iterations
            elif (next_error[0] < error[0]) and (next_x >= 0).all():
                residuals[:] = next_residuals[:]
                X0[:] = next_x[:]
                error[0] = next_error[0]
                step += (max_step - step) * step_adjustment
            else:
                step = 1/step
                step += (step + 1/max_step)
                step = 1/step
            if step <= 1e-10:
                return X0, error, iterations
        else:
            return X0, error, iterations
        
    @staticmethod
    @njit
    def _solve_cg(
        A_val,
        A_col_idx,
        A_row_ptr, 
        b,
        max_iterations, # sqrt(n) for n x n system
        target_error, # 1.0e-6
        X0,
        threads,
    ):
        #Reference: https://repository.lsu.edu/cgi/viewcontent.cgi?article=1254&context=honors_etd

        x = X0.copy()
        r = b.copy()
        m = np.empty(1, dtype=np.float64)
        m[0] = _square_sum_vector(r, threads) # f(x:vector) = x'*x
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
                threads,
                ) # scalar_product = p'*A*p
            _add_product(x, alpha[0], p, threads) # f(x: vector, y: scalar, z:vector): x += y * z
            #_subtract_product_of_product(
            #    r, 
            #    alpha[0], 
            #    A_val, 
            #    A_col_idx, 
            #    A_row_ptr,
            #    p, 
            #    threads,
            #) # f(x:vector, y:scalar, z:array, k:vector): x -= y * z * k
            _recalc_residuals_jit(r, A_val, A_col_idx, A_row_ptr, b, x)
            m_last[0] = m[0]
            m[0] = _square_sum_vector(r, threads)
            beta[0] = m[0] / m_last[0]
            _multiply_and_add(
                p, 
                r, 
                beta[0], 
                threads,
            ) # f(x:vector, y:vector, z:scalar): x = y + z * x
            error = np.sqrt(_square_sum_vector(r, threads) 
                            / _square_sum_vector(b, threads)
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
        max_iterations, # sqrt(n) for n x n system
        target_error, # 1.0e-6
        X0,
        threads,
    ):
        #Reference: https://repository.lsu.edu/cgi/viewcontent.cgi?article=1254&context=honors_etd

        x = X0.copy()
        r = b.copy()
        m = np.empty(1, dtype=np.float64)
        m[0] = _scalar_product(
                r, 
                P_val, 
                P_col_idx, 
                P_row_ptr, 
                threads,
                ) # scalar_product = p'*A*p
        m_last = np.empty(1, dtype=np.float64)
        p = r.copy()
        p[:] = _vector_array_multiply(
            p, 
            P_val, 
            P_col_idx, 
            P_row_ptr, 
            threads,
            ) # f(v, A): v = A*v
        alpha = np.empty(1, dtype=np.float64)
        beta = np.empty(1, dtype=np.float64)
        iteration = 0
        for _ in range(max_iterations):
            #print()
            #print("X", x)
            #print("R", r)
            #print("P", p)
            #print("Alpha=",alpha," Beta=",beta," M=", m)
            iteration += 1
            alpha[0] = m[0] / _scalar_product(
                p, 
                A_val, 
                A_col_idx, 
                A_row_ptr, 
                threads,
                ) # scalar_product = p'*A*p
            _add_product(x, alpha[0], p, threads) # f(x: vector, y: scalar, z:vector): x += y * z
            _recalc_residuals_jit(r, A_val, A_col_idx, A_row_ptr, b, x)
            m_last[0] = m[0]
            m[0] = _scalar_product(
                r, 
                P_val, 
                P_col_idx, 
                P_row_ptr, 
                threads,
                ) # scalar_product = p'*A*p
            beta[0] = m[0] / m_last[0]
            p[:] = _multiply_array_and_add(
                p, 
                r, 
                beta[0],
                P_val, 
                P_col_idx, 
                P_row_ptr,
                threads,
            ) # f(x:vector, y:vector, z:scalar, A:array): x = A * y + z * x
            error = np.sqrt(_square_sum_vector(r, threads) 
                            / _square_sum_vector(b, threads)
            )
            if error <= target_error:
                return x, error, iteration

        #print(x)
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
                A_stop = A_row_ptr[i+1]
            else:
                A_stop = A_val.size
            for A_linear_index in range(A_start, A_stop):
                j = A_col_idx[A_linear_index]
                partial_sum[w] += v[i] * A_val[A_linear_index] * v[j]
    return partial_sum.sum()


@njit(parallel=True)
def _add_product(v, x, u, threads) :
    # f(v: vector, x: scalar, u:vector): v += x * u
    n = v.size

    for w in prange(threads):
        thread_start = w * n // threads
        thread_end = (w + 1) * n // threads
        for i in range(thread_start, thread_end):
            v[i] += x * u[i]


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
                A_stop = A_row_ptr[i+1]
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


@njit#(parallel=True)
def _vector_array_multiply(
    p, 
    val, 
    col_idx, 
    row_ptr, 
    threads,
    ): # f(v, A): v = A*v
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
    ): # f(u:vector, v:vector, x:scalar, A:array): x = A * v + x * u
        new_p = _vector_array_multiply(
            v, 
            val, 
            col_idx, 
            row_ptr, 
            threads,
            )
        new_p += x * u
        return new_p


@njit(parallel=True)
def _calc_residuals_jit(val, col_idx, row_ptr, condensed_b, X):

    #residuals = np.zeros(condensed_b.size, dtype=np.float64)
    residuals = condensed_b.copy()

    for row in range(row_ptr.size):
        start = row_ptr[row]
        if row < (row_ptr.size - 1):
            stop = row_ptr[row + 1]
        else:
            stop = val.size

        for index in range(start, stop):
            v = val[index]
            column = col_idx[index]
            residuals[row] -= v * X[column]
    return residuals


@njit(parallel=True)
def _recalc_residuals_jit(r, val, col_idx, row_ptr, condensed_b, X):

    #residuals = np.zeros(condensed_b.size, dtype=np.float64)
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
def _update_resdiuals_and_error_parallel(
    next_residuals, 
    next_error, 
    val, 
    col_idx, 
    row_ptr, 
    condensed_b, 
    X, 
    threads,
):
    next_error_partial = np.zeros(threads, dtype=np.float32)
    next_residuals.fill(0)
    rows_n = row_ptr.size

    for w in prange(threads):
        thread_start = w * rows_n // threads
        thread_end = (w + 1) * rows_n // threads
        for row in range(thread_start, thread_end):
            start = row_ptr[row]
            if row < (row_ptr.size - 1):
                stop = row_ptr[row + 1]
            else:
                stop = val.size

            residual = np.float32(0)
            for index in range(start, stop):
                v = val[index]
                column = col_idx[index]
                x_val = X[column]
                residual += v * x_val
            residual -= condensed_b[row]
            next_residuals[row] = residual
            next_error_partial[w] += (residual**2)
    next_error[0] = next_error_partial.sum() / X.size


@njit(parallel=True)
def _calc_residuals_jit_parallel(val, col_idx, row_ptr, condensed_b, X, threads):

    residuals = np.zeros(condensed_b.size, dtype=np.float32)
    rows_n = row_ptr.size

    for w in prange(threads):
        thread_start = w * rows_n // threads
        thread_end = (w + 1) * rows_n // threads
        for row in range(thread_start, thread_end):
            start = row_ptr[row]
            if row < (row_ptr.size - 1):
                stop = row_ptr[row + 1]
            else:
                stop = val.size

            residual = np.float32(0)
            for index in range(start, stop):
                v = val[index]
                column = col_idx[index]
                x_val = X[column]
                residual += v * x_val
            residual -= condensed_b[row]
            residuals[row] = residual
    return residuals


@njit(parallel=True)
def _calc_next_x(next_x, residuals, step, diag, X0, threads):
    length = next_x.size
    for w in prange(threads):
        thread_start = w * length // threads
        thread_end = (w + 1) * length // threads
        for i in range(thread_start, thread_end):
            next_x[i] = residuals[i]
            next_x[i] *= step
            next_x[i] /= diag[i]
            next_x[i] *= -1.
            next_x[i] += X0[i]


@njit
def _getitem(val, col_idx, row_ptr, row, col):

    start, stop = _row_bounds(row, val, row_ptr)
    for i in range(start, stop):
        if col_idx[i] == col:
            return val[i]
    return np.float32(0)


@njit
def _row_bounds(row, val, row_ptr):
    rows = row_ptr.size
    if row <= (rows - 2):
        start = row_ptr[row]
        stop = row_ptr[row+1]
    elif row == rows - 1:
        start = row_ptr[row]
        stop = val.size
    return start, stop


@njit
def _get_diagonal_preconditioner(
    A_val, 
    A_col_idx, 
    A_row_ptr, 
    threads,
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
