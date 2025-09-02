import unittest
from lb.models import *
from lb.data.lb_data import LBData
import math
import numpy as np
from lb.evaluation.evaluation import LBEvaluation
from lb.models.lb_dwave_qubo_unary_uniform_truck_capacity import LBDWAVEQUBOUnaryUniformTruckCapacity
from dwave.samplers import SimulatedAnnealingSampler


class MyTestCase(unittest.TestCase):
    def test_qubo_gurobi(self):
        number_of_trucks = 5
        capacity_per_truck = 10
        total_load_per_truck = 100
        data = LBData.from_random(number_of_trucks, capacity_per_truck, total_load_per_truck)
        model = LBGurobiQUBOUnaryUniformTruckCapacity(data)
        model.model.setParam("MIPGap", 0.5)
        model.solve()
        solution = model.solution()
        evaluation_object = LBEvaluation(solution, data)
        optimal_solution_value = number_of_trucks * total_load_per_truck**2
        self.assertTrue(evaluation_object.check_solution())
        self.assertLessEqual(evaluation_object.get_objective(), 2*optimal_solution_value)

    def test_cqm_gurobi(self):
        number_of_trucks = 5
        capacity_per_truck = 10
        total_load_per_truck = 100
        data = LBData.from_random(number_of_trucks, capacity_per_truck, total_load_per_truck)
        model = LBGurobiCQMUnaryUniformTruckCapacity(data)
        model.model.setParam("MIPGap", 1E-5)
        model.solve()
        solution = model.solution()
        evaluation_object = LBEvaluation(solution, data)
        optimal_solution_value = number_of_trucks * total_load_per_truck**2
        self.assertTrue(evaluation_object.check_solution())
        self.assertLessEqual(evaluation_object.get_objective(), optimal_solution_value)

    def test_qubo_dwave(self):
        number_of_trucks = 5
        capacity_per_truck = 10
        total_load_per_truck = 100
        data = LBData.from_random(number_of_trucks, capacity_per_truck, total_load_per_truck)
        # Create gurobi model
        model = LBDWAVEQUBOUnaryUniformTruckCapacity(data)
        # Solve problem
        solve_func = SimulatedAnnealingSampler().sample_qubo
        model.solve(solve_func, num_reads=100)

        solution = model.solution()
        evaluation_object = LBEvaluation(solution, data)
        optimal_solution_value = number_of_trucks * total_load_per_truck**2
        self.assertTrue(evaluation_object.check_solution())
        self.assertLessEqual(evaluation_object.get_objective(), 2*optimal_solution_value)

    def test_cqm_dwave(self):
        number_of_trucks = 5
        capacity_per_truck = 10
        total_load_per_truck = 100
        data = LBData.from_random(number_of_trucks, capacity_per_truck, total_load_per_truck)
        # Create gurobi model
        model = LBDWAVECQMUnaryUniformTruckCapacity(data)
        # Solve problem
        solve_func = SimulatedAnnealingSampler().sample
        model.solve(solve_func, num_reads=100)

        solution = model.solution()
        evaluation_object = LBEvaluation(solution, data)
        optimal_solution_value = number_of_trucks * total_load_per_truck**2
        self.assertTrue(evaluation_object.check_solution())
        self.assertLessEqual(evaluation_object.get_objective(), 2*optimal_solution_value)


if __name__ == '__main__':
    unittest.main()
