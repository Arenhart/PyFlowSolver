# PyFlowSolver

CFD laminar flow solver for porous media and pore networks. It provides **two
independent flow models**:

- **Fast-Laplacian** — solves `∇·(k∇p) = 0` on the voxel grid with EDT-estimated conductivities, based on Arns Approximation. Cheap and robust; gives velocity and pressure fields.
- **Stokes** — pore-resolved steady viscous flow via a projection / fractional-step method on a MAC staggered grid. Gives velocity and pressure fields.

**Note**: A fast-Laplacian solution can be used as a warm start for Stokes to cut the computation time up to 60%, depending on pore complexity and error tolerance. This is the default behaviour of the Stoke solver, but can be disabled by passing fast_laplacian_guess=False

## Installation

```bash
pip install -e .                 # editable install
pip install -r requirements.txt  # pinned dependencies
```

## Geometry convention

A geometry is a 3D `numpy` array `(w, h, d)`. Any **positive** value is pore, `0` is solid. By default flow is driven **along z**: `z = 0` is the inlet (pressure = 1) and `z = max` is the outlet (pressure = 0).

---
## 1. Workflows

### 1.1 Fast-Laplacian approximation

Assigns per-voxel conductivity from the pore geometry and solves as a Darcy system, as described in Arns and Adler, 2018 (DOI: https://doi.org/10.1103/PhysRevE.97.023303). The conductivities are calculated in a way that the final velocity fields approximate a true Stokes flow, but only requires the simpler Darcy flow algorithms to solve.

Example:
```python
from pyflowsolver.volumeManager import VolumeManager
from pyflowsolver.darcySolver import DarcySolver

volume = ...                       # 3D array, >0 = pore, 0 = solid
poremap = (volume > 0) * 100.0
scale = 0.02                       # voxel size (single value or (dx, dy, dz))

vm = VolumeManager(poremap, scale=scale)
vm.convert_pore_volume_to_laplacian_conductivity()
a_sparse, b = vm.get_sparse_system_jit()

solver = DarcySolver()
solver.set_linear_system(a_sparse, b)
solver.generate_preconditioner(preconditioner="inverse_diagonal")
solution, error, iterations = solver.solve_pcg()

pressure = vm.ravel_sparse_solution(solution)   # pressure field, grid-shaped
k_eff = vm.get_conductivity(pressure)           # effective (Darcy) conductivity
print("Darcy effective conductivity:", k_eff)
```

---

### 1.2 Stokes solver

`StokesSolver` takes the **raw voxel array** and solves to steady state, returning
the MAC velocity components and the pressure. Regular (z-driven) geometry only.

```python
import numpy as np
from pyflowsolver.stokesSolver import StokesSolver

volume = ...                       # 3D array, >0 = pore, 0 = solid

solver = StokesSolver(
    volume,
    scale=1.0,                     # isotropic voxel size
    viscosity=1.0,                 # kinematic viscosity nu = mu / rho
    density=1.0,                   # rho
    target_error=1e-6,             # steady-state velocity-change tolerance
    max_iterations=20000,
    fast_laplacian_guess=True,     # already true by Default
)
result = solver.solve()

u, v, w = result["u"], result["v"], result["w"]   # MAC face velocities
p = result["p"]                                   # cell-centered pressure
print(f"converged={result['converged']} "
      f"iterations={result['iterations']} residual={result['residual']:.2e}")

# Cell-centered velocity + volumetric flux through a mid cross-section:
w_center = (w[:, :, :-1] + w[:, :, 1:]) / 2         # axial velocity at cell centers
mid_length = volume.shape[2] // 2
flux = w_center[:, :, mid_length].sum() * (solver.scale[0] * solver.scale[1])  
print("volumetric flux Q:", flux)
```



