import numpy as np
from numba import njit, prange, typed

from pyflowsolver.constants import SOLID, PORE, INLET, OUTLET

class VolumeManager():

    def __init__(self, volume, boundary_volume=None):
        """
        volume: A float ndarray with the local conductivity of each voxel
        boundary_volume: A uint8 ndarray, same shape as volume, with the 
            following value convention:
            SOLID = 0
            PORE = 1
            INLET = 2
            OUTLET = 3
        """
        self.volume = volume
        self.boundary_volume = boundary_volume
        self.nulls_count = np.empty(volume.size, dtype=int)
        if boundary_volume is None:
            self._calc_null_counts(self.volume, self.nulls_count)
        else:
            self._calc_null_counts_irregular(self.boundary_volume, self.nulls_count)
        self._generate_neighbours_dict()
        self.nonzeros = self.volume.size - self.nulls_count[-1]
        self.len_x = 1
        self.len_y = 1
        self.len_z = 1

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
            "val" : val_array,
            "col_idx" : col_idx_array,
            "row_ptr" : row_ptr_array,
        }

        return sparse_array, condensed_b
    
    def get_sparse_system(self):
        w, h, d = self.volume.shape
        
        val_array = np.zeros(self.nonzeros * 7)
        col_idx_array = np.zeros(self.nonzeros * 7, dtype=int)
        row_ptr_array = np.zeros(self.nonzeros, dtype=int)
        condensed_b = np.zeros(self.nonzeros)
        
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

        val_array.resize(vals_n)
        col_idx_array.resize(vals_n)

        sparse_array = {
            "val" : val_array,
            "col_idx" : col_idx_array,
            "row_ptr" : row_ptr_array,
        }

        return sparse_array, condensed_b


    def get_sparse_system_jit(self):
        
        val_array = np.zeros(self.nonzeros * 7, dtype=float)
        col_idx_array = np.zeros(self.nonzeros * 7, dtype=int)
        row_ptr_array = np.zeros(self.nonzeros, dtype=int)
        condensed_b = np.zeros(self.nonzeros, dtype=float)

        if self.boundary_volume is not None:
            irregular_boundary = True
            boundary_volume=self.boundary_volume
        else:
            irregular_boundary = False
            boundary_volume=np.zeros((1, 1, 1), dtype=np.uint8)

        val_array, col_idx_array = _jit_sparse_system_extraction(
            val_array,
            col_idx_array,
            row_ptr_array,
            condensed_b,
            conductivity_volume=self.volume,
            nulls_count=self.nulls_count,
            irregular_boundary=irregular_boundary,
            boundary_volume=boundary_volume,
        )

        sparse_array = {
            "val" : val_array,
            "col_idx" : col_idx_array,
            "row_ptr" : row_ptr_array,
        }

        return sparse_array, condensed_b

    def _unravel(self, x, y, z):
        _, h, d = self.volume.shape
        i = z + y * d + x * d * h

        output = i - self.nulls_count[i]

        return output
    
    @staticmethod
    @njit
    def _calc_null_counts(volume, nulls_count):
        running_zeros = 0
        i = 0
        w, h, d = volume.shape
        for x in range(w):
            for y in range(h):
                for z in range(d):
                    center_c = volume[x, y, z]
                    if center_c > 0:
                        nulls_count[i] = running_zeros
                        i += 1
                    else:
                        running_zeros += 1
                        nulls_count[i] = running_zeros
                        i += 1
    

    @staticmethod
    @njit
    def _calc_null_counts_irregular(boundary_volume, nulls_count):
        running_zeros = 0
        i = 0
        w, h, d = boundary_volume.shape
        for x in range(w):
            for y in range(h):
                for z in range(d):
                    center_element = boundary_volume[x, y, z]
                    if center_element == PORE:
                        nulls_count[i] = running_zeros
                        i += 1
                    else:
                        running_zeros += 1
                        nulls_count[i] = running_zeros
                        i += 1

    
    def ravel_dense_solution(self, solution):
        raveled_solution = np.zeros_like(self.volume)
        w, h, d = raveled_solution.shape
        for i, (x, y, z) in enumerate(
            (a,b,c) for a in range(w) for b in range(h) for c in range(d)
            ):
            raveled_solution[x, y, z] = solution[i]

        return raveled_solution
    
    def ravel_sparse_solution(self, solution):
        raveled_solution = np.zeros_like(self.volume)
        w, h, d = raveled_solution.shape
        i = 0
        for x, y, z in ((a,b,c) for a in range(w) for b in range(h) for c in range(d)):
            if self.boundary_volume is None:
                if self.volume[x, y, z] > np.float32(0):
                    raveled_solution[x, y, z] = solution[i]
                    i += 1
            elif self.boundary_volume is not None:
                if self.boundary_volume[x, y, z] == PORE:
                    raveled_solution[x, y, z] = solution[i]
                    i += 1
                if self.boundary_volume[x, y, z] == INLET:
                    raveled_solution[x, y, z] = np.float32(1)

        return raveled_solution

    def get_velocity_array(self):
        velocity_array = np.zeros((self.nonzeros, 3), dtype=np.float32)
        return velocity_array
    
    def get_pressure_array(self):
        pressure_array = np.zeros(self.nonzeros, dtype=np.float32)
        return pressure_array
    
    def get_laplacian_poisson(self, boundaries=None):
        
        w, h, d = self.volume.shape
        dx = self.len_x
        dy = self.len_y
        dz = self.len_z
        len_array = np.array((dx, dy, dz), dtype=np.int32)

        val_array = np.zeros(self.nonzeros * 7)
        col_idx_array = np.zeros(self.nonzeros * 7, dtype=int)
        row_ptr_array = np.zeros(self.nonzeros, dtype=int)

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

            total_c = 0

            if z_min or z_max:
                total_c -= 2 / dz

            center_i = self._unravel(x, y, z)
            row_ptr_array[center_i] = vals_n
            for neighbour in neighbours:
                neighbour_c = self.volume[tuple(coords + neighbour)]
                if neighbour_c == 0: continue
                face_c = np.float32(-1 / (len_array*neighbour.abs()).sum())
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
            "val" : val_array,
            "col_idx" : col_idx_array,
            "row_ptr" : row_ptr_array,
        }

@njit
def _jit_sparse_system_extraction(
            val_array,
            col_idx_array,
            row_ptr_array,
            condensed_b,
            conductivity_volume,
            nulls_count,
            irregular_boundary,
            boundary_volume,
        ):
    vals_n = np.uint16(0)
    shape = conductivity_volume.shape
    w = np.uint16(shape[0])
    h = np.uint16(shape[1])
    d = np.uint16(shape[2])

    for x in range(w): 
        for y in range(h):
            for z in range(d):
                center_c = conductivity_volume[x, y, z]

                if center_c == 0:
                    continue

                if irregular_boundary:
                    if boundary_volume[x, y, z] != PORE:
                        continue

                x_min = (x == 0)
                x_max = (x == w - 1)
                y_min = (y == 0)
                y_max = (y == h - 1)
                z_min = (z == 0)
                z_max = (z == d - 1)

                x_index = 1 - x_min*1 + x_max*1
                y_index = 1 - y_min*1 + y_max*1
                z_index = 1 - z_min*1 + z_max*1

                neighbours = _get_neighbours(x_index, y_index, z_index)

                total_c = np.float32(0)

                if not irregular_boundary:
                    if z_min:
                        total_c += 2 * center_c
                        condensed_b[_unravel(x, y, z, h, d, nulls_count)] = -(2 * center_c)
                    elif z_max:
                        total_c += 2 * center_c

                center_i = _unravel(x, y, z, h, d, nulls_count)
                row_ptr_array[center_i] = vals_n

                for neighbour in neighbours:
                    neighbour_c = conductivity_volume[x+neighbour[0], y+neighbour[1], z+neighbour[2]]
                    if irregular_boundary:
                        neighbour_element = boundary_volume[x+neighbour[0], y+neighbour[1], z+neighbour[2]]
                    else: #not irregular_boundary
                        if neighbour_c == 0:
                            neighbour_element = SOLID
                        else:
                            neighbour_element = PORE
                    
                    if neighbour_element == SOLID:
                        continue

                    elif neighbour_element == PORE:
                        face_c = np.float32(2 / (1 / center_c + 1 / neighbour_c))
                        total_c += np.float32(face_c)
                        neighbour_i = _unravel(x+neighbour[0], y+neighbour[1], z+neighbour[2], h, d, nulls_count)
                        val_array[vals_n] = face_c
                        col_idx_array[vals_n] = neighbour_i
                        vals_n += 1

                    elif neighbour_element in [INLET, OUTLET]:
                        face_c = np.float32(2 * center_c)
                        total_c += np.float32(face_c)
                        #center element is always PORE
                        if neighbour_element == INLET:
                            condensed_b[_unravel(x, y, z, h, d, nulls_count)] = -face_c

                val_array[vals_n] = -total_c
                col_idx_array[vals_n] = _unravel(x, y, z, h, d, nulls_count)
                vals_n += 1

    val_array = val_array[:vals_n]
    col_idx_array = col_idx_array[:vals_n]

    return val_array, col_idx_array


@njit
def make_unravel_function(h, d, nulls_count):

    def unravel(x, y, z):
        i = np.uint16(z) + np.uint16(y) * d + np.uint16(x) * d * h

        output = i - nulls_count[i]

        return output
    
    return unravel


@njit
def _unravel(x, y, z, h, d, nulls_count):

    i = np.uint16(z) + np.uint16(y) * d + np.uint16(x) * d * h

    output = i - nulls_count[i]

    return output


@njit
def _get_neighbours(x, y, z):
    #x_max, y_max, z_max
    if (x == 0) and(y == 0) and(z == 0):
        return list( ((1, 0, 0), (0, 1, 0), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 0) and(y == 0) and(z == 1):
        return list( ((1, 0, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 0) and(y == 0) and(z == 2):
        return list( ((1, 0, 0), (0, 1, 0), (0, 0, -1)) )
    #x_max, y_max, z_max
    elif (x == 0) and(y == 1) and(z == 0):
        return list( ((1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 0) and(y == 1) and(z == 1):
        return list( ((1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 0) and(y == 1) and(z == 2):
        return list( ((1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1)) )
    #x_max, y_max, z_max
    elif (x == 0) and(y == 2) and(z == 0):
        return list( ((1, 0, 0), (0, -1, 0), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 0) and(y == 2) and(z == 1):
        return list( ((1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 0) and(y == 2) and(z == 2):
        return list( ((1, 0, 0), (0, -1, 0), (0, 0, -1)) )
    #x_max, y_max, z_max
    elif (x == 1) and(y == 0) and(z == 0):
        return list( ((-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 1) and(y == 0) and(z == 1):
        return list( ((-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 1) and(y == 0) and(z == 2):
        return list( ((-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, -1)) )
    #x_max, y_max, z_max
    elif (x == 1) and(y == 1) and(z == 0):
        return list( ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 1) and(y == 1) and(z == 1):
        return list( ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 1) and(y == 1) and(z == 2):
        return list( ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1)) )
    #x_max, y_max, z_max
    elif (x == 1) and(y == 2) and(z == 0):
        return list( ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 1) and(y == 2) and(z == 1):
        return list( ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 1) and(y == 2) and(z == 2):
        return list( ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 0, -1)) )
    #x_max, y_max, z_max
    elif (x == 2) and(y == 0) and(z == 0):
        return list( ((-1, 0, 0), (0, 1, 0), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 2) and(y == 0) and(z == 1):
        return list( ((-1, 0, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 2) and(y == 0) and(z == 2):
        return list( ((-1, 0, 0), (0, 1, 0), (0, 0, -1)) )
    #x_max, y_max, z_max
    elif (x == 2) and(y == 1) and(z == 0):
        return list( ((-1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 2) and(y == 1) and(z == 1):
        return list( ((-1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 2) and(y == 1) and(z == 2):
        return list( ((-1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1)) )
    #x_max, y_max, z_max
    elif (x == 2) and(y == 2) and(z == 0):
        return list( ((-1, 0, 0), (0, -1, 0), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 2) and(y == 2) and(z == 1):
        return list( ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)) )
    #x_max, y_max, z_max
    elif (x == 2) and(y == 2) and(z == 2):
        return list( ((-1, 0, 0), (0, -1, 0), (0, 0, -1)) )