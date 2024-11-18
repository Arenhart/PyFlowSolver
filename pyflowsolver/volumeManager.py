import numpy as np
import scipy as sc
from numba import njit, prange, typed

from pyflowsolver.constants import SOLID, PORE, INLET, OUTLET
from pyflowsolver.fastLaplacian import fast_laplacian_volume_generator

class VolumeManager():

    def __init__(self, volume, scale=1, boundary_volume=None):
        """
        volume: A float ndarray with the local conductivity or a porosity map,
            if porosity_map, should run convert_pore_volume_to_laplacian_conductivity
            afterwards
        dimension: Single number or tuple for the voxel dimension
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
        self.filter_connected_volume()
        if boundary_volume is None:
            self._calc_null_counts(self.volume, self.nulls_count)
        else:
            self._calc_null_counts_irregular(self.boundary_volume, self.nulls_count)
        self._generate_neighbours_dict()
        self.nonzeros = self.volume.size - self.nulls_count[-1]
        try: 
            iter(scale)
        except:
            self.scale = (
                scale,
            ) * 3
        else:
            self.scale = np.float32(scale[:3])


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


    def convert_pore_volume_to_laplacian_conductivity(self, porosity_map=False):
        if not porosity_map:
            self.volume = (self.boundary_volume >= 1)*100
            self.volume = fast_laplacian_volume_generator(
                self.volume, 
                self.scale,
                closed_border=False, 
                )
        else:
            raise("Not implemented yet")


    def filter_connected_volume(self):
        
        if self.boundary_volume is not None:
            labeled_volume, _ = sc.ndimage.label(self.boundary_volume > 0)
            self._filter_connected_volume_irregular(self.boundary_volume, labeled_volume)
        else: 
            labeled_volume, _ = sc.ndimage.label(self.volume > 0)
            self._filter_connected_volume(self.volume, labeled_volume)


    @staticmethod
    @njit
    def _filter_connected_volume_irregular(boundary_volume, labels):
        w, h, d = boundary_volume.shape
        inlet_set = set()
        outlet_set = set()
        for i in range(w):
            for j in range(h):
                for k in range(d):
                    if boundary_volume[i, j, k] == INLET:
                        label = labels[i, j, k]
                        if label != 0:
                            inlet_set.add(label)
                    elif boundary_volume[i, j, k] == OUTLET:
                        label = labels[i, j, k]
                        if label != 0:
                            outlet_set.add(label)
        connected_labels = inlet_set.intersection(outlet_set)
        for i in range(w):
            for j in range(h):
                for k in range(d):
                    if labels[i, j, k] not in connected_labels:
                        boundary_volume[i,j,k] = 0


    @staticmethod
    @njit
    def _filter_connected_volume(pore_volume, labels):
        w, h, d = labels.shape
        inlet_set = set(np.unique(labels[:, :, 0]))
        outlet_set = set(np.unique(labels[:, :, -1]))
        connected_labels = inlet_set.intersection(outlet_set)

        for i in range(w):
            for j in range(h):
                for k in range(d):
                    if labels[i, j, k] not in connected_labels:
                        pore_volume[i,j,k] = 0


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
        # Unfinished function
        w, h, d = self.volume.shape
        dx, dy, dz = self.scale
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

    def get_conductivity(self, pressure_volume):
        conductivity = self._get_flow(
            pressure_volume, 
            self.boundary_volume, 
            conductivity_volume=self.volume, 
            scale=self.scale,
        )
        return conductivity

    @staticmethod
    @njit
    def _get_flow(
            pressure_volume, 
            boundary_volume, 
            conductivity_volume, 
            scale, 
            pressure_difference=1,
            ):
        w, h, d = pressure_volume.shape
        area_0 = scale[1] * scale[2] / scale[0]
        area_1 = scale[0] * scale[2] / scale[1]
        area_2 = scale[0] * scale[1] / scale[2]
        in_flow = np.float64(0)
        out_flow = np.float64(0)
        for x1 in range(w-1):
            for y1 in range(h-1):
                for z1 in range(d-1):
                    center_boundary = boundary_volume[x1, y1, z1]
                    if center_boundary == SOLID:
                        continue
                    for (x2, y2, z2, area) in ((x1+1, y1, z1, area_0), (x1, y1+1, z1, area_1), (x1, y1, z1+1, area_2)):
                        neighbour_boundary = boundary_volume[x2, y2, z2]
                        center_pressure = pressure_volume[x1, y1, z1]
                        neighbour_pressure = pressure_volume[x2, y2, z2]
                        if neighbour_boundary == SOLID:
                            continue
                        # Flow from neighbour to center
                        if ((center_boundary == PORE and neighbour_boundary == INLET)
                                or (center_boundary == OUTLET and neighbour_boundary == PORE)):
                            delta_p = neighbour_pressure - center_pressure
                        # Flow from center to neighbour
                        elif ((center_boundary == INLET and neighbour_boundary == PORE)
                                or (center_boundary == PORE and neighbour_boundary == OUTLET)):
                            delta_p = center_pressure - neighbour_pressure
                        else:
                            continue
                        if center_boundary == PORE:
                            conductivity = 2 * conductivity_volume[x1, y1, z1]
                        elif neighbour_boundary == PORE:
                            conductivity = 2 * conductivity_volume[x2, y2, z2]

                        flow = delta_p * conductivity * area

                        if (center_boundary == INLET) or (neighbour_boundary) == INLET:
                            in_flow += flow
                        else:
                            out_flow += flow
        relative_flow = (in_flow + out_flow) / 2 # flow/pressure_difference
        flow = relative_flow * pressure_difference
        return flow


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