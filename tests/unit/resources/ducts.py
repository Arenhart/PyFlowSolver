"""Analytic geometry generators for solver verification.

`make_circular_duct` builds a straight circular pipe aligned with the z axis:
a circular fluid cross-section (in the x-y plane) extruded along z. The Darcy
convention drives flow z=0 (inlet) -> z=max (outlet), so once the Stokes solver
is complete the steady z-velocity profile can be compared against the analytic
Hagen-Poiseuille solution for laminar pipe flow:

    w(r) = (G / (4 mu)) * (R^2 - r^2)          (parabolic profile)
    Q    = pi * R^4 * G / (8 mu)               (volumetric flow rate)

with dynamic viscosity mu = rho * nu, pipe radius R, and pressure gradient
G = dp/dz. This module only produces the geometry; the analytic comparison
lives in the test that exercises the finished solver.
"""

import numpy as np


def make_circular_duct(radius, length, margin=2):
    """Return a float voxel volume for a circular duct aligned with z.

    radius: pipe radius in voxels.
    length: number of voxels along the flow (z) direction.
    margin: solid voxels padding the circular cross-section on each side.

    The returned array is 1.0 inside the pipe and 0.0 (solid wall) outside,
    with shape (n, n, length) where n = ceil(2*radius) + 2*margin. Feed it to
    `VolumeManager` as a conductivity volume; fluid cells are `volume > 0`.
    """
    n = int(np.ceil(2 * radius)) + 2 * margin
    volume = np.zeros((n, n, length), dtype=np.float64)

    center = (n - 1) / 2.0
    xx, yy = np.mgrid[0:n, 0:n]
    disk = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    volume[disk, :] = 1.0

    return volume
