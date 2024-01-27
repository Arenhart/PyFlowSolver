import numpy as np

class SparseArray():

    def __init__(self, val, col_idx, row_ptr):
        self.val = val
        self.col_idx = col_idx
        self.row_ptr = row_ptr

        self.size = val.size
        self.shape = np.array((
            row_ptr.size,
            col_idx.max() + 1,
            ), 
            dtype=int,
        )
        self.rows = self.shape[0]
        self.columns = self.shape[1]
        self._sort_sparse_array()

    def row_iterator(self):
        for row, (start, stop) in enumerate(
            zip(
                self.row_ptr[:-1], 
                self.row_ptr[1:],
            )):
            yield row, start, stop
        else:
            yield self.rows - 1, self.row_ptr[-1], self.val.size,
    
    def row_bounds(self, row):
        if row <= (self.rows - 2):
            start = self.row_ptr[row]
            stop = self.row_ptr[row+1]
        elif row == self.rows - 1:
            start = self.row_ptr[row]
            stop = self.val.size
        return start, stop

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.val[key]
        if isinstance(key, tuple):
            row = key[0]
            col = key[1]
            val = 0
            start, stop = self.row_bounds(row)
            for i in range(start, stop):
                if self.col_idx[i] == col:
                    val = self.val[i]
                    break
            return val
        
    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.val[key] = value
        if isinstance(key, tuple):
            row = key[0]
            col = key[1]
            start, stop = self.row_bounds(row)
            for i in range(start, stop):
                if self.col_idx[i] == col:
                    self.val[i] = value
                    break
            else:
                raise IndexError("Cannot assign value to uninitiated index")
            
    def _sort_sparse_array(self):
        for _, start, stop in self.row_iterator():
            val = self.val[start:stop]
            col_idx = self.col_idx[start:stop]
            col_idx, val = zip(*sorted(zip(col_idx, val)))
            self.val[start:stop] = val
            self.col_idx[start:stop] = col_idx
                

            
