import numpy as np

class VolumeManager():

    def __init__(self, volume):
        self.volume = volume
        self.nulls_count = None
        self._generate_neighbours_dict()

    def _generate_neighbours_dict(self):
        self.neighbours_dict = {}
        for x, y, z in ((a,b,c) for a in range(3) for b in range(3) for c in range(3)):
            x_min = (x == 0)
            x_max = (x == 2)
            y_min = (y == 0)
            y_max = (y == 2)
            z_min = (z == 0)
            z_max = (z == 2)
            key = (x_min, x_max, y_min, y_max, z_min, z_max)
            neighbours = []
            if x_min is False: neighbours.append(np.array((-1,0,0), dtype=int))
            if x_max is False: neighbours.append(np.array((1,0,0), dtype=int))
            if y_min is False: neighbours.append(np.array((0,-1,0), dtype=int))
            if y_max is False: neighbours.append(np.array((0,1,0), dtype=int))
            if z_min is False: neighbours.append(np.array((0,0,-1), dtype=int))
            if z_max is False: neighbours.append(np.array((0,0,1), dtype=int))
            self.neighbours_dict[key] = neighbours

    def get_linear_system(self):

        A = np.zeros((self.volume.size, self.volume.size), dtype = np.float32)
        b = np.zeros(self.volume.size, dtype = np.float32)

        i = 0
        w, h, d = self.volume.shape
        neighbour_displacement_template = np.array((h*d, d, 1), dtype=int)

        for x, y, z in ((a, b, c) for a in range(w) for b in range(h) for c in range(d)):
            coords = np.array((x, y, z))
            center_c = self.volume[tuple(coords)]

            if center_c == 0:
                A[i, i] = 1
                i += 1
                continue

            x_min = (x == 0)
            x_max = (x == w - 1)
            y_min = (y == 0)
            y_max = (y == h - 1)
            z_min = (z == 0)
            z_max = (z == d - 1)

            key = (x_min, x_max, y_min, y_max, z_min, z_max)
            neighbours = self.neighbours_dict[key]

            total_c = np.float32(0)

            if z_min:
                total_c += 2 * center_c
                b[i] = -(2 * center_c)
            elif z_max:
                total_c += 2 * center_c

            for neighbour in neighbours:
                neighbour_c = self.volume[tuple(coords + neighbour)]
                if neighbour_c == 0: continue
                face_c = 2 / (1 / center_c + 1 / neighbour_c)
                total_c += np.float32(face_c)
                neighbour_displacement = (neighbour_displacement_template * neighbour).sum()
                A[i, i + neighbour_displacement] = face_c
            A[i,i] = - total_c

            i += 1

        return A, b
    
    def get_sparse_system_from_dense(self, A, b):

        diag = np.diagonal(A)
        diag_nulls_count = np.zeros(diag.shape, dtype=int)
        diag_nulls_count[0] = diag[0] > 0
        for i in range(1, diag_nulls_count.size):
            diag_nulls_count[i] = diag_nulls_count[i-1] + (diag[i] > 0)
        nonzero_n = np.count_nonzero(diag < 0)
        val_array = np.zeros(nonzero_n * 6)
        col_idx_array = np.zeros(nonzero_n * 6, dtype=int)
        row_ptr_array = np.zeros(nonzero_n, dtype=int)
        condensed_b = np.zeros(nonzero_n)

        sparse_i = 0
        vals_n = 0

        vals_n = 0
        for dense_i in range(diag.size):
            if diag[dense_i] > 0:
                continue

            row_ptr_array[sparse_i] = vals_n

            for dense_j in range(diag.size):
                if A[dense_i, dense_j] != 0:
                    val_array[vals_n] = A[dense_i, dense_j]
                    real_col = dense_j - diag_nulls_count[dense_j]
                    col_idx_array[vals_n] = real_col
                    condensed_b[sparse_i] = b[dense_i]
                    vals_n += 1

            sparse_i += 1
        val_array.resize(vals_n)
        col_idx_array.resize(vals_n)

        sparse_array = {
            "val": val_array,
            "col_idx": col_idx_array,
            "row_ptr": row_ptr_array,
        }

        return sparse_array, condensed_b
    
    def get_sparse_system(self):
        nonzeros = np.count_nonzero(self.volume)
        self.nulls_count = np.empty(self.volume.size, dtype=int)
        running_zeros = 0
        i = 0
        w, h, d = self.volume.shape
        for x, y, z in ((a,b,c) for a in range(w) for b in range(h) for c in range(d)):
            center_c = self.volume[x, y, z]
            if center_c > 0:
                self.nulls_count[i] = running_zeros
                i += 1
            else:
                running_zeros += 1
                self.nulls_count[i] = running_zeros
                i += 1
        
        val_array = np.zeros(nonzeros * 7)
        col_idx_array = np.zeros(nonzeros * 7, dtype=int)
        row_ptr_array = np.zeros(nonzeros, dtype=int)
        condensed_b = np.zeros(nonzeros)
        
        sparse_i = 0
        vals_n = 0

        for x, y, z in ((a,b,c) for a in range(w) for b in range(h) for c in range(d)):
            coords = np.array((x, y, z))
            center_c = self.volume[tuple(coords)]

            if center_c == 0:
                continue

            x_min = (x == 0)
            x_max = (x == w - 1)
            y_min = (y == 0)
            y_max = (y == h - 1)
            z_min = (z == 0)
            z_max = (z == d - 1)

            key = (x_min, x_max, y_min, y_max, z_min, z_max)
            neighbours = self.neighbours_dict[key]

            total_c = np.float32(0)

            if z_min:
                total_c += 2 * center_c
                condensed_b[self._unravel(x, y, z)] = -(2 * center_c)
            elif z_max:
                total_c += 2 * center_c

            center_i = self._unravel(x, y, z)
            row_ptr_array[center_i] = vals_n
            for neighbour in neighbours:
                neighbour_c = self.volume[tuple(coords + neighbour)]
                if neighbour_c == 0: continue
                face_c = np.float32(2 / (1 / center_c + 1 / neighbour_c))
                total_c += np.float32(face_c)
                neighbour_i = self._unravel(*tuple(coords + neighbour))
                val_array[vals_n] = face_c
                col_idx_array[vals_n] = neighbour_i
                vals_n += 1
            val_array[vals_n] = -total_c
            col_idx_array[vals_n] = self._unravel(*tuple(coords))
            vals_n += 1
            i += 1

        val_array.resize(vals_n)
        col_idx_array.resize(vals_n)

        sparse_array = {
        "val": val_array,
        "col_idx": col_idx_array,
        "row_ptr": row_ptr_array,
        }

        self._sort_sparse_array(sparse_array)

        return sparse_array, condensed_b

    def _unravel(self, x, y, z):
        _, h, d = self.volume.shape
        i = z + y * d + x * d * h

        output = i - self.nulls_count[i]

        return output
    
    def _sort_sparse_array(self, sparse_array):
        for start, stop in self.row_indexes_iterator(sparse_array):
            val = sparse_array["val"][start:stop]
            col_idx = sparse_array["col_idx"][start:stop]
            col_idx, val = zip(*sorted(zip(col_idx, val)))
            sparse_array["val"][start:stop] = val
            sparse_array["col_idx"][start:stop] = col_idx


    def row_indexes_iterator(self, sparse_array):
        for start, stop in zip(sparse_array["row_ptr"][:-1], sparse_array["row_ptr"][1:]):
            yield start, stop
        else:
            yield sparse_array["row_ptr"][-1], sparse_array["val"].size
