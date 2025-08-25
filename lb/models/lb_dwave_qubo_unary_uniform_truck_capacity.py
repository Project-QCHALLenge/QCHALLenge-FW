import itertools
import time

from lb.models.lb_qubo_uniform_truck_capacity import LBQUBOUnaryUniformTruckCapacity
import os
from lb.data.lb_data import LBData
from lb.models.lb_abstract_model import LBAbstractModel
import numpy as np


class LBDWAVEQUBOUnaryUniformTruckCapacity(LBAbstractModel):

    def __init__(self, data: LBData):
        super().__init__(data)
        self.Q = None
        self.sample_set = None
        self.build_model()
        self.runtime = -1

    def solve(self, solve_func, **params):
        num_reads = params["num_reads"]
        start_time = time.time()
        self.sample_set = solve_func(self.Q, num_reads=num_reads)
        finish_time = time.time()
        self.runtime = finish_time - start_time

    def solution(self):
        solution_string_as_np_array = np.array(list(self.sample_set.first.sample.values()))
        number_of_units = int(len(solution_string_as_np_array) / self.data.number_of_trucks)
        solution_matrix = solution_string_as_np_array.reshape(self.data.number_of_trucks, number_of_units)
        all_indices = list(itertools.product(range(solution_matrix.shape[0]), range(solution_matrix.shape[1])))
        return {
            "solution" : { f"x_{int(x[0])}_{int(x[1])}" : int(solution_matrix[x]) for x in all_indices},
            "energy": float(self.sample_set.first.energy),
            "runtime": self.runtime
        }

    def build_model(self):
        self.Q, _, _ = LBQUBOUnaryUniformTruckCapacity(self.data).assemble()

    def _create_variables(self):
        pass

    def _create_data_structures(self):
        pass

    def set_initial_solution(self, solution):
        pass
