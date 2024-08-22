from pyflowsolver.sparseArray import SparseArray

class VelocityArray(SparseArray):

    def __init__(self, val, col_idx, row_ptr):
        self.super(val, col_idx, row_ptr)