
import pytest
import numpy as np

val = np.array((3,2,2,1,2,2,1,3,2,4,4,1,2,3), dtype = np.float32)
col_idx = np.array((0,1,1,2,5,1,2,2,3,4,1,4,4,5), dtype=int)
row_ptr = np.array((0,2,5,7,10,12), dtype=int)

from src.sparseArray import SparseArray

sparse_array = SparseArray(val, col_idx, row_ptr)

def test_get_item():
    val_1 = sparse_array[0, 0]
    val_2 = sparse_array[0, 5]
    val_3 = sparse_array[5, 5]
    val_4 = sparse_array[4]
    assert(val_1 == np.float32(3))
    assert(val_2 == np.float32(0))
    assert(val_3 == np.float32(3))
    assert(val_4 == np.float32(2))

def test_set_item():
    sparse_array[0,0] = np.float32(5)
    assert(sparse_array[0,0] == np.float32(5))
    assert(sparse_array[0,0] == sparse_array[0])
    sparse_array[1] = np.float32(6)
    assert(sparse_array[1] == np.float32(6))
    assert(sparse_array[1] == sparse_array[0,1])
    with pytest.raises(IndexError) as excinfo:  
        sparse_array[0,2] = np.float32(5)  
    assert str(excinfo.value) == "Cannot assign value to uninitiated index"

def test_array_shape():
    assert(np.array_equal(sparse_array.shape,np.array((6,6), dtype=int)))

def test_array_size():
    assert(sparse_array.size == int(14))

def test_row_iterator():
    rows = [
        (0, 0, 2),
        (1, 2, 5),
        (2, 5, 7),
        (3, 7, 10),
        (4, 10, 12),
        (5, 12, 14),
    ]

    for row, start, stop in sparse_array.row_iterator():
        row_t, start_t, stop_t = rows.pop(0)
        assert(row == row_t)
        assert(start == start_t)
        assert(stop == stop_t)


