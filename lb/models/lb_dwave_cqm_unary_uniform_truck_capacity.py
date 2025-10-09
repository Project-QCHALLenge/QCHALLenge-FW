import itertools

import dimod
import numpy as np
from lb.models.lb_cqm_uniform_truck_capacity import LBCQMUniformTruckCapacity
import os
from lb.data.lb_data import LBData
from lb.models.lb_abstract_model import LBAbstractModel
from dimod import CQM, Binary
import time
from abstract.models.abstract_model import AbstractModel


class LBDWAVECQMUnaryUniformTruckCapacity(LBAbstractModel, AbstractModel):

    def __init__(self, data: LBData):
        super().__init__(data)
        self.Q, self.A, self.rhs = LBCQMUniformTruckCapacity(self.data).assemble()
        self.variables = None
        self.sample_set = None
        self.model = CQM()
        self.trucks = range(self.data.number_of_trucks)
        self.units = range(self.data.number_of_units)
        self._create_data_structures()
        self._create_variables()
        self.build_model()
        self.runtime = -1

    def solve(self, solve_func, **params):
        num_reads = params["num_reads"]
        bqm, _ = dimod.cqm_to_bqm(self.model)
        # Needs to be done the right way, no access to quantum hardware or whatever is needed
        start_time = time.time()
        self.sample_set = solve_func(bqm, num_reads=num_reads)
        finish_time = time.time()
        self.runtime = finish_time - start_time

    def solution(self):
        solution_string_as_np_array = np.array(list(self.sample_set.first.sample.values()))
        number_of_units = int(len(solution_string_as_np_array) / self.data.number_of_trucks)
        solution_matrix = solution_string_as_np_array.reshape(self.data.number_of_trucks, number_of_units)
        all_indices = list(itertools.product(range(solution_matrix.shape[0]), range(solution_matrix.shape[1])))
        return {
            "solution": {f"x_{int(x[0])}_{int(x[1])}": int(solution_matrix[x]) for x in all_indices},
            "energy": float(self.sample_set.first.energy),
            "runtime": self.runtime
        }

    def build_model(self):
        # Adds Constraints, no matrix support so constraints are added row by row
        for i in range(self.rhs.size):
            self.model.add_constraint(self.A[i, :] @ self.decision_variables == self.rhs[i])

        self.model.set_objective(self.decision_variables.T @ self.Q @ self.decision_variables)

    def _create_variables(self):
        self.decision_variables = np.array([Binary(index) for index in self.valid_indices])

    def _create_data_structures(self):
        self.valid_indices = itertools.product(self.trucks, self.units)

    def set_initial_solution(self, solution):
        pass

