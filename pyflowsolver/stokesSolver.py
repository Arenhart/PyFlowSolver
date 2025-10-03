import numpy as np

from pyflowsolver.solver import Solver

ZERO = np.float64(0.0)
X = 0
Y = 1
Z = 2
class StokesSolver(Solver):
    def __init__(self, volume_manager):
        self.default_params = {
            "max_step" : 1/4,
            "step_adjustment" : 1/4,
            "initial_step" : 1/8,
            "max_iterations" : 5000,
            "target_error" : 1e-07,
            "viscosity" : 1,
        }
        self.volume_manager = volume_manager
        self.create_velocity_arrays()


    def create_velocity_arrays(self):
        volume = self.volume_manager.volume
        w, h, d = volume.shape

        self.x_vel_nulls_count_array = np.zeros((w-1)*h*d, dtype=np.uint32)
        missing_n = 0
        for z in range(d):
            for y in range(h):
                for x in range(1, w):
                    if (volume[x, y, z] == 0) or (volume[(x-1), y, z] == 0):
                        missing_n += 1
                    index = (x-1) + (y*(w-1)) + (z*(w-1)*h)
                    self.x_vel_nulls_count_array[index] = missing_n

        self.y_vel_nulls_count_array = np.zeros(w*(h-1)*d, dtype=np.uint32)
        missing_n = 0
        for z in range(d):
            for y in range(1, h):
                for x in range(w):
                    if (volume[x, y, z] == 0) or (volume[x, (y-1), z] == 0):
                        missing_n += 1
                    index = x + ((y-1)*w) + (z*w*(h-1))
                    self.y_vel_nulls_count_array[index] = missing_n

        self.z_vel_nulls_count_array = np.zeros(w*h*(d+1), dtype=np.uint32)
        missing_n = 0
        for z in range(d+1):
            for y in range(h):
                for x in range(w):
                    if z == 0:
                        if (volume[x, y, z] == 0):
                            missing_n += 1
                    if z == d:
                        if (volume[x, y, z-1] == 0):
                            missing_n += 1
                    else:
                        if (volume[x, y, z] == 0) or (volume[x, y, z-1] == 0):
                            missing_n += 1
                    index = x + (y*w) + (z*w*h)
                    self.z_vel_nulls_count_array[index] = missing_n

        self.pressure_nulls_count_array = self.volume_manager.nulls_count

        self.x_vel_array = np.zeros(
            (w-1)*h*d - self.x_vel_nulls_count_array[-1], 
            dtype=np.float64,
        )
        self.x_star_vel_array = self.x_vel_array.copy()
        self.y_vel_array = np.zeros(
            w*(h-1)*d - self.y_vel_nulls_count_array[-1], 
            dtype=np.float64,
        )
        self.y_star_vel_array = self.y_vel_array.copy()
        self.z_vel_array = np.zeros(
            w*h*(d+1) - self.z_vel_nulls_count_array[-1], 
            dtype=np.float64,
        )
        self.z_star_vel_array = self.z_vel_array.copy()
        self.pressure_array = np.zeros(
            self.volume_manager.volume.size - self.volume_manager.nulls_count[-1],
            dtype=np.float64,
        )


    def get_voxel_info(self, x, y, z, star=False):
        self.volume_manager.volume
        w, h, d = self.volume_manager.volume.shape
        if self.volume_manager.volume[x, y, z] == 0:
            result = {
                "x_vel_in": ZERO,
                "x_vel_out": ZERO,
                "y_vel_in": ZERO,
                "y_vel_out": ZERO,
                "z_vel_in": ZERO,
                "z_vel_out": ZERO,
                "pressure": ZERO,
            }
            return result

        x_vel_in = self.get_x_vel_in(x, y, z, star)
        x_vel_out = self.get_x_vel_in(x+1, y, z, star)
        y_vel_in = self.get_y_vel_in(x, y, z, star)
        y_vel_out = self.get_y_vel_in(x, y+1, z, star)
        z_vel_in = self.get_z_vel_in(x, y, z, star)
        z_vel_out = self.get_z_vel_in(x, y, z+1, star)
        pressure = self.get_pressure(x, y, z)
        result = {
            "x_vel_in": x_vel_in,
            "x_vel_out": x_vel_out,
            "y_vel_in": y_vel_in,
            "y_vel_out": y_vel_out,
            "z_vel_in": z_vel_in,
            "z_vel_out": z_vel_out,
            "pressure": pressure,
        }
        return result


    def get_x_vel_in(self, x, y, z, star=False):
        w, h, _ = self.volume_manager.volume.shape
        if (x == 0) or (x == w) or self.volume_manager.volume[x, y, z] == 0:
            return ZERO
        else:
            index = (x-1) + y*(w-1) + z*(w-1)*h
            index -= self.x_vel_nulls_count_array[index]
            if not star:
                return self.x_vel_array[index]
            else:
                return self.x_star_vel_array[index]
        
    def get_y_vel_in(self, x, y, z, star=False):
        w, h, _ = self.volume_manager.volume.shape
        if (y == 0) or (y == h) or self.volume_manager.volume[x, y, z] == 0:
            return ZERO
        else:
            index = x + (y-1)*w + z*w*(h-1)
            index -= self.y_vel_nulls_count_array[index]
            if not star:
                return self.y_vel_array[index]
            else:
                return self.y_star_vel_array[index]
        
    def get_z_vel_in(self, x, y, z, star=False):
        w, h, d = self.volume_manager.volume.shape
        
        if z < 0:
            return self.get_z_vel_in(x, y, (z+1))
        elif z > d:
            return self.get_z_vel_in(x, y, (z-1))
        elif ((z < d and self.volume_manager.volume[x, y, z] == 0)
            or (z == d and self.volume_manager.volume[x, y, z-1] == 0)):
            return ZERO
        else:
            index = x + y*w + z*w*h
            index -= self.z_vel_nulls_count_array[index]
            if not star:
                return self.z_vel_array[index]
            else:
                return self.z_star_vel_array[index]
        
    def get_pressure(self, x, y, z):
        w, h, _ = self.volume_manager.volume.shape

        index = x + y*w + z*w*h
        index -= self.pressure_nulls_count_array[index]
        return self.pressure_array[index]
    

    def set_x_vel_in(self, x, y, z, new_vel, star=False):
        w, h, _ = self.volume_manager.volume.shape
        if (
            (x == 0) 
            or (x == w) 
            or self.volume_manager.volume[x, y, z] == 0
            or self.volume_manager.volume[x-1, y, z] == 0
        ):
            raise ValueError(
                f"Trying to set x velocity in boudary at coords [{x},{y},{z}]"
                )
        else:
            index = (x-1) + y*(w-1) + z*(w-1)*h
            index -= self.x_vel_nulls_count_array[index]
            if not star:
                self.x_vel_array[index] = new_vel
            else:
                self.x_star_vel_array[index] = new_vel
        
    def set_y_vel_in(self, x, y, z, new_vel, star=False):
        w, h, _ = self.volume_manager.volume.shape
        if (
            (y == 0) 
            or (y == h) 
            or self.volume_manager.volume[x, y, z] == 0
            or self.volume_manager.volume[x, y-1, z] == 0
        ):
            raise ValueError(
                f"Trying to set y velocity in boudary at coords [{x},{y},{z}]"
                )
        else:
            index = x + (y-1)*w + z*w*(h-1)
            index -= self.y_vel_nulls_count_array[index]
            if not star:
                self.y_vel_array[index] = new_vel
            else:
                self.y_star_vel_array[index] = new_vel
        
    def set_z_vel_in(self, x, y, z, new_vel, star=False):
        w, h, d = self.volume_manager.volume.shape
        
        if z < 0:
            raise ValueError(
                f"Trying to set z velocity in boudary at coords [{x},{y},{z}]"
                )
        elif z > d:
            raise ValueError(
                f"Trying to set z velocity in boudary at coords [{x},{y},{z}]"
                )
        elif ((z < d and self.volume_manager.volume[x, y, z] == 0)
            or (z == d and self.volume_manager.volume[x, y, z-1] == 0)):
            raise ValueError(
                f"Trying to set z velocity in boudary at coords [{x},{y},{z}]"
                )
        else:
            index = x + y*w + z*w*h
            index -= self.z_vel_nulls_count_array[index]
            if not star:
                self.z_vel_array[index] = new_vel
            else:
                self.z_star_vel_array[index] = new_vel
        
    def set_pressure(self, x, y, z, new_pressure):
        w, h, _ = self.volume_manager.volume.shape

        if (self.volume_manager.volume[x, y, z] == 0):
            raise ValueError(
                f"Trying to set pressure in solid at coords [{x},{y},{z}]"
                )
        index = x + y*w + z*w*h
        index -= self.pressure_nulls_count_array[index]
        self.pressure_array[index] = new_pressure


    def update_vel_star(self):
        delta_t = self.default_params["initial_step"]
        viscosity = self.default_params["viscosity"]
        self.x_star_vel_array[:] = self.x_vel_array.copy()
        self.y_star_vel_array[:] = self.y_vel_array.copy()
        self.z_star_vel_array[:] = self.z_vel_array.copy()
        volume = self.volume_manager.volume
        scale = self.volume_manager.scale
        w, h, d = volume.shape

        ux = lambda x, y, z : self.get_x_vel_in(x, y, z)
        uy = lambda x, y, z : self.get_y_vel_in(x, y, z)
        uz = lambda x, y, z : self.get_z_vel_in(x, y, z)

        get_vel_funcs = {
            X: self.get_x_vel_in,
            Y: self.get_y_vel_in,
            Z: self.get_z_vel_in,
        }
        def d2(direction_ax, diff_ax, i, j, k):
            # computes second derivative d(2f) / d(ax)2
            f = get_vel_funcs[direction_ax]
            i0 = i1 = i
            j0 = j1 = j
            k0 = k1 = k
            if diff_ax == X:
                i0 =  i - 1
                i1 = i + 1
            elif diff_ax == Y:
                j0 = j - 1
                j1 = j + 1
            elif diff_ax == Z:
                k0 = k - 1
                k1 = k + 1

            return (f(i0, j0, k0) - 2*f(i, j, k) + f(i1, j1, k1)) / (scale[diff_ax]**2)

        def ud(direction_ax, diff_ax, i, j, k):
            # The differential direction (diff_ax) always equals the undiferentiated velocity direction
            # direction_ax refers to the velocity vector field direction beign updated
            # computes velocity times first derivative u_ax * df/d(ax)
            f = get_vel_funcs[direction_ax]
            i0 = i1 = i
            j0 = j1 = j
            k0 = k1 = k
            if diff_ax == X:
                i0 = i + 1
                i1 = i - 1
            elif diff_ax == Y:
                j0 = j + 1
                j1 = j - 1
            elif diff_ax == Z:
                k0 = k + 1
                k1 = k - 1

            if direction_ax == diff_ax:
                u = f(i, j, k)
            else:
                di0 = di1 = di2 = di3 = i
                dj0 = dj1 = dj2 = dj3 = j
                dk0 = dk1 = dk2 = dk3 = k

                if direction_ax == X and diff_ax == Y:
                    di1 -= 1
                    di2 -= 1
                    dj2 += 1
                    dj3 += 1
                elif direction_ax == Y and diff_ax == X:
                    di1 += 1
                    di2 += 1
                    dj2 -= 1
                    dj3 -= 1
                elif direction_ax == X and diff_ax == Z:
                    di1 -= 1
                    di2 -= 1
                    dk2 += 1
                    dk3 += 1
                elif direction_ax == Z and diff_ax == X:
                    di1 += 1
                    di2 += 1
                    dk2 -= 1
                    dk3 -= 1
                elif direction_ax == Y and diff_ax == Z:
                    dj1 -= 1
                    dj2 -= 1
                    dk2 += 1
                    dk3 += 1
                elif direction_ax == Z and diff_ax == Y:
                    dj1 += 1
                    dj2 += 1
                    dk2 -= 1
                    dk3 -= 1

                u = (
                    f(di0, dj0, dk0)
                    + f(di1, dj1, dk1)
                    + f(di2, dj2, dk2)
                    + f(di3, dj3, dk3)
                    ) / 4
                
            return u * (f(i0, j0, k0) + f(i1, j1, k1)) / (2*scale[diff_ax])

        for z in range(d+1):
            for y in range(h):
                for x in range(w):
                    for direction in (X, Y, Z):
                        new_u = get_vel_funcs[X](z, y, x)
                        new_u += delta_t * (
                            viscosity * (d2(direction, X, x, y, z) + d2(direction, Y, x, y, z) + d2(direction, Z, x, y, z))
                            - (ud(direction, X, x, y, z) + ud(direction, Y, x, y, z) + ud(direction, Z, x, y, z))
                            )
                        try:
                            self.set_x_vel_in(
                                x, 
                                y, 
                                z, 
                                new_u, 
                                star=True
                            )
                        except ValueError:
                            continue


    def predictor_step(self):
        pass

    def poisson_step(self):
        pass

    def corrector_step(self):
        pass