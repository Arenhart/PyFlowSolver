import numpy as np

from pyflowsolver.solver import Solver

ZERO = np.float64(0.0)
class StokesSolver(Solver):
    def __init__(self, volume_manager):
        self.default_params = {
            "max_step" : 1/4,
            "step_adjustment" : 1/4,
            "initial_step" : 1/8,
            "max_iterations" : 5000,
            "target_error" : 1e-07,
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
        self.y_vel_array = np.zeros(
            w*(h-1)*d - self.y_vel_nulls_count_array[-1], 
            dtype=np.float64,
        )
        self.z_vel_array = np.zeros(
            w*h*(d+1) - self.z_vel_nulls_count_array[-1], 
            dtype=np.float64,
        )
        self.pressure_array = np.zeros(
            self.volume_manager.volume.size - self.volume_manager.nulls_count[-1],
            dtype=np.float64,
        )


    def get_voxel_info(self, x, y, z):
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

        x_vel_in = self.get_x_vel_in(x, y, z)
        x_vel_out = self.get_x_vel_in(x+1, y, z)
        y_vel_in = self.get_y_vel_in(x, y, z)
        y_vel_out = self.get_y_vel_in(x, y+1, z)
        z_vel_in = self.get_z_vel_in(x, y, z)
        z_vel_out = self.get_z_vel_in(x, y, z+1)
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


    def get_x_vel_in(self, x, y, z):
        w, h, _ = self.volume_manager.volume.shape
        if (x == 0) or (x == w) or self.volume_manager.volume[x, y, z] == 0:
            return ZERO
        else:
            index = (x-1) + y*(w-1) + z*(w-1)*h
            index -= self.x_vel_nulls_count_array[index]
            return self.x_vel_array[index]
        
    def get_y_vel_in(self, x, y, z):
        w, h, _ = self.volume_manager.volume.shape
        if (y == 0) or (y == h) or self.volume_manager.volume[x, y, z] == 0:
            return ZERO
        else:
            index = x + (y-1)*w + z*w*(h-1)
            index -= self.y_vel_nulls_count_array[index]
            return self.y_vel_array[index]
        
    def get_z_vel_in(self, x, y, z):
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
            return self.z_vel_array[index]
        
    def get_pressure(self, x, y, z):
        w, h, _ = self.volume_manager.volume.shape

        index = x + y*w + z*w*h
        index -= self.pressure_nulls_count_array[index]
        return self.pressure_array[index]
    

    def set_x_vel_in(self, x, y, z, new_vel):
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
            self.x_vel_array[index] = new_vel
        
    def set_y_vel_in(self, x, y, z, new_vel):
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
            self.y_vel_array[index] = new_vel
        
    def set_z_vel_in(self, x, y, z, new_vel):
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
            self.z_vel_array[index] = new_vel
        
    def set_pressure(self, x, y, z, new_pressure):
        w, h, _ = self.volume_manager.volume.shape

        if (self.volume_manager.volume[x, y, z] == 0):
            raise ValueError(
                f"Trying to set pressure in solid at coords [{x},{y},{z}]"
                )
        index = x + y*w + z*w*h
        index -= self.pressure_nulls_count_array[index]
        self.pressure_array[index] = new_pressure


    def predictor_step(self):
        pass

    def poisson_step(self):
        pass

    def corrector_step(self):
        pass