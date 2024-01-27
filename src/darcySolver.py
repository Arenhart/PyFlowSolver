import numpy as np

from src.solver import Solver

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

