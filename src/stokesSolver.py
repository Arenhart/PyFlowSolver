import numpy as np

from src.solver import Solver

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


    def create_velocity_arrays(self):
        pass

    def predictor_step(self):
        pass

    def poisson_step(self):
        pass

    def corrector_step(self):
        pass