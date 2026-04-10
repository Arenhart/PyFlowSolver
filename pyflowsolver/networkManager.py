import numpy as np
from numba import njit, prange, typed

from pyflowsolver.sparseArray import SparseArray

class NetworkManager():

    def __init__(self, conn, cond, inlets, outlets):
        self.conn = conn
        self.cond = cond
        self.inlets = inlets
        self.outlets = outlets
        self.a_sparse_array = None
        self.b_array = None
        self.mid_to_total_indexes = None


    def generate_sparse_system(self):
        (
            sparse_val, 
            sparse_col_idx, 
            sparse_row_ptr, 
            b, 
            mid_to_total_indexes,
        ) = self._calc_sparse_system(
            self.conn, 
            self.cond, 
            self.inlets, 
            self.outlets,
        )
        self.a_sparse_array = {
            "val" : sparse_val,
            "col_idx" : sparse_col_idx,
            "row_ptr" : sparse_row_ptr,
        }
        self.b_array = b
        self.mid_to_total_indexes = mid_to_total_indexes


    def get_sparse_system(self, scipy_format=False):
        if scipy_format:
            sparse_row_ptr = np.empty(
                self.a_sparse_array["row_ptr"].size + 1, 
                dtype=self.a_sparse_array["row_ptr"].dtype,
                )
            sparse_row_ptr[:-1] = self.a_sparse_array["row_ptr"]
            sparse_row_ptr[-1] = self.a_sparse_array["val"].size
            scipy_sparse_array = {
            "val" : self.a_sparse_array["val"],
            "col_idx" : self.a_sparse_array["col_idx"],
            "row_ptr" : sparse_row_ptr,
        }
            return scipy_sparse_array, self.b_array
        else:
            return self.a_sparse_array, self.b_array


    def get_pressure_list(self, x):

        pressure = np.zeros(self.inlets.size, dtype=np.float64)
        pressure[self.mid_to_total_indexes] = x * np.float64(101325)
        for i in range(self.inlets.size):
            if self.inlets[i] == 1:
                pressure[i] = np.float64(101325)
            elif self.outlets[i] == 1:
                pressure[i] = np.float64(0)
        return pressure


    def get_flow_rate(self, pressures):
        inlet_flow_total = np.float64(0.0)
        outlet_flow_total = np.float64(0.0)
        border_pore = np.logical_or(self.inlets, self.outlets)
        throats_n = self.cond.size

        flow = np.zeros(throats_n, dtype=np.float64)
        delta_p = np.zeros(throats_n, dtype=np.float64)
        inlet_flow = np.zeros(throats_n, dtype=np.float64)
        outlet_flow = np.zeros(throats_n, dtype=np.float64)
        for throat in range(throats_n):
            p0 = self.conn[throat, 0]
            p1 = self.conn[throat, 1]
            c = self.cond[throat]
            delta_p[throat] = np.abs(p0 - p1)
            flow[throat] = delta_p[throat] * c

        border_pore = np.logical_or(self.inlets, self.outlets)
        for throat in range(throats_n):
            p0 = self.conn[throat, 0]
            p1 = self.conn[throat, 1]
            c = self.cond[throat]
            if self.inlets[p0] and (not border_pore[p1]):
                inlet_flow_total += c * (
                    np.float64(101325.0) - pressures[p1]
                )
                inlet_flow[throat] = c * (
                    np.float64(101325.0) - pressures[p1]
                )
            if self.inlets[p1] and (not border_pore[p0]):
                inlet_flow_total += c * (
                    np.float64(101325.0) - pressures[p0]
                )
                inlet_flow[throat] = c * (
                    np.float64(101325.0) - pressures[p0]
                )
            if self.outlets[p0] and (not border_pore[p1]):
                outlet_flow_total += c * (pressures[p1])
                outlet_flow[throat] = c * (pressures[p1])
            if self.outlets[p1] and (not border_pore[p0]):
                outlet_flow_total += c * (pressures[p0])
                outlet_flow[throat] = c * (pressures[p0])

        flow_rate = (outlet_flow_total + inlet_flow_total) / 2
        return flow_rate


    @staticmethod
    @njit
    def _calc_sparse_system(conn, cond, inlets, outlets):
        # network must have only connected pores
        # conn array(n, 2)
        # assumes inlet pressure = 1 and outlet pressure = 0

        inlets = np.logical_and(inlets, np.logical_not(outlets))
        border = np.logical_or(inlets, outlets)

        n_p_total = inlets.size
        n_p_in = inlets.sum()
        n_p_out = outlets.sum()
        n_p_mid = n_p_total - n_p_in - n_p_out
        n_t = cond.size

        mid_to_total_indexes = np.zeros((n_p_mid), dtype=np.int32)
        total_to_mid_indexes = np.zeros((n_p_total), dtype=np.int32)

        pore_index_filled = 0
        for p in range(n_p_total):
            if border[p] == 0:
                mid_to_total_indexes[pore_index_filled] = p
                total_to_mid_indexes[p] = pore_index_filled
                pore_index_filled += 1
        if pore_index_filled != mid_to_total_indexes.size:
            raise Exception

        sparse_row_counter = np.zeros((n_p_mid), dtype=np.int32)
        n_mid_t = 0
        for t in range(n_t):
            i = conn[t, 0]
            j = conn[t, 1]
            if (border[i] == 0) and (border[j] == 0):
                n_mid_t += 1
                sparse_row_counter[total_to_mid_indexes[i]] += 1
                sparse_row_counter[total_to_mid_indexes[j]] += 1

        sparse_val = np.zeros((n_p_mid + 2 * n_mid_t), dtype=np.float64)
        sparse_col_idx = np.ones((sparse_val.size), dtype=np.int32) * -1
        sparse_row_ptr = np.zeros((n_p_mid), dtype=np.int32)
        sparse_col_idx[0] = 0
        for i in range(1, n_p_mid):
            sparse_row_ptr[i] = sparse_row_ptr[i - 1] + sparse_row_counter[i - 1] + 1
            sparse_col_idx[sparse_row_ptr[i]] = i

        b = np.zeros(n_p_mid, dtype=np.float64)

        for t in range(n_t):
            conn_0 = conn[t, 0]
            conn_1 = conn[t, 1]
            conductance = cond[t]

            for i, j in ((conn_0, conn_1), (conn_1, conn_0)):
                if border[i] == 1:
                    continue
                i_mid = total_to_mid_indexes[i]
                row_ptr_start = sparse_row_ptr[i_mid]
                sparse_val[row_ptr_start] -= conductance

                if inlets[j] == 1:
                    b[i_mid] -= conductance
                elif border[j] == 0:
                    j_mid = total_to_mid_indexes[j]
                    # target column is j_mid
                    # first check if column is already occupied
                    if (i_mid + 1) < sparse_row_ptr.size:
                        row_ptr_end = sparse_row_ptr[i_mid + 1]
                    else:
                        row_ptr_end = sparse_val.size

                    found = False
                    for linear_index in range(row_ptr_start, row_ptr_end):
                        if sparse_col_idx[linear_index] == j_mid:
                            sparse_val[linear_index] += conductance
                            found = True
                            break
                    if not found:
                        local_index = sparse_row_counter[i_mid]
                        linear_index = row_ptr_start + local_index
                        if (local_index <= 0) or (sparse_val[linear_index] != 0):
                            raise Exception
                        sparse_col_idx[linear_index] = j_mid
                        sparse_val[linear_index] = conductance
                        sparse_row_counter[i_mid] -= 1
                        found = True
                    if not found:
                        raise Exception

        # sparse cleanup
        cumulative_nulls = 0
        for i in range(sparse_row_ptr.size):
            row_ptr_start = sparse_row_ptr[i]
            if (i + 1) < sparse_row_ptr.size:
                row_ptr_stop = sparse_row_ptr[i + 1]
            else:
                row_ptr_stop = sparse_val.size
            nulls = (sparse_col_idx[row_ptr_start:row_ptr_stop] == -1).sum()
            filled = row_ptr_stop - row_ptr_start - nulls
            sort_index = np.argsort(sparse_col_idx[row_ptr_start:row_ptr_stop])
            sorted_vals = sparse_val[row_ptr_start:row_ptr_stop][sort_index]
            sorted_col_idx = sparse_col_idx[row_ptr_start:row_ptr_stop][sort_index]
            compacted_start = row_ptr_start - cumulative_nulls
            compacted_end = compacted_start + filled
            sparse_val[compacted_start:compacted_end] = sorted_vals[nulls:]
            sparse_col_idx[compacted_start:compacted_end] = sorted_col_idx[nulls:]
            if i >= 1:
                sparse_row_ptr[i] = compacted_start
            cumulative_nulls += nulls
        if cumulative_nulls > 0:
            sparse_val = sparse_val[: sparse_val.size - cumulative_nulls]
            sparse_col_idx = sparse_col_idx[: sparse_col_idx.size - cumulative_nulls]

        return sparse_val, sparse_col_idx, sparse_row_ptr, b, mid_to_total_indexes

