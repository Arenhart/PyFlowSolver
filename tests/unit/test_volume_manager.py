import sys
import os

import numpy as np
import pytest

from src.volumeManager import VolumeManager
from tests.unit.resources.templates import (
    A_template, 
    b_template,
    sparse_array_template,
)


volume = np.ones((4,4,4), dtype=int)
volume[1:3, 1:3, 1] = 0
volume[1:3, 1:3, 2] = 2
volume_manager = VolumeManager(volume)

def test_get_linear_system():
    A, b = volume_manager.get_linear_system()
    np.testing.assert_array_equal(A, A_template)
    np.testing.assert_array_equal(b, b_template)

def test_get_sparse_system():
    A, b = volume_manager.get_linear_system()
    sparse_array = volume_manager.get_sparse_system(A, b)
    np.testing.assert_array_equal(sparse_array["val"], sparse_array_template["val"])
    np.testing.assert_array_equal(sparse_array["col_idx"], sparse_array_template["col_idx"])
    np.testing.assert_array_equal(sparse_array["row_ptr"], sparse_array_template["row_ptr"])



