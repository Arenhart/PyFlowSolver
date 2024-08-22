import numpy as np
from numba import njit

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

        for _ in range(param("max_iterations")):
            next_x = X0 - residuals * step / diag
            next_residuals = self.calc_residuals(A, b, next_x)
            next_error = (next_residuals**2).sum()/X0.size
            print(next_x, step, next_error, error)
            if next_error <= param("target_error"):
                return next_x
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
            return X0


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

    def solve_jit(self, A, b, **params):

        max_step = params.get("max_step", self.default_params["max_step"])
        step_adjustment = params.get("step_adjustment", self.default_params["step_adjustment"])
        initial_step = params.get("initial_step", self.default_params["initial_step"])
        max_iterations = params.get("max_iterations", self.default_params["max_iterations"])
        target_error = params.get("target_error", self.default_params["target_error"])

        X0 = np.linspace(1, 0, num=b.size, dtype=np.float32)
        print(A.val)
        print(A.col_idx)
        print(A.row_ptr)
        print(b)

        X0 = self._solve_jit(
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
        print(X0)
        return X0
        

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
        for i in range(A_rows):
            diag[i] = _getitem(A_val, A_col_idx, A_row_ptr, i, i)
        for _ in range(max_iterations):
            next_x[:] = residuals[:]
            next_x *= step
            next_x /= diag
            next_x *= np.float32(-1)
            next_x += X0
            next_residuals = _calc_residuals_jit(A_val, A_col_idx, A_row_ptr, b, next_x)
            next_error = (next_residuals**2).sum()/X0.size
            if next_error <= target_error:
                return next_x
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
                return X0
        else:
            return X0


@njit
def _calc_residuals_jit(val, col_idx, row_ptr, condensed_b, X):

    residuals = np.zeros(condensed_b.size, dtype=np.float32)

    for row in range(row_ptr.size):
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
