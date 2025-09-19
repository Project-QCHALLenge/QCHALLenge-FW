import unittest
from lb.models import *
from lb.data.lb_data import LBData
import math
import numpy as np
from lb.evaluation.evaluation import LBEvaluation
from lb.models.lb_dwave_qubo_unary_uniform_truck_capacity import LBDWAVEQUBOUnaryUniformTruckCapacity
from dwave.samplers import SimulatedAnnealingSampler
import gurobipy as gp
from gurobipy import GRB
import dimod


class TestLBModels(unittest.TestCase):

    @staticmethod
    def create_gurobi_model_from_Q(Q):
        model = gp.Model("QUBO")
        variables = model.addVars(range(Q.shape[0]), vtype=GRB.BINARY)
        x = gp.MVar.fromlist(list(variables.values()))
        model.setObjective(x.T @ Q @ x, GRB.MINIMIZE)
        return model

    @staticmethod
    def solve_function(Q, *args, **kwargs):
        if type(Q) == dimod.BinaryQuadraticModel:
            Q = Q.to_numpy_matrix(variable_order=Q.variables)
        model = TestLBModels.create_gurobi_model_from_Q(Q)
        model.setParam("TimeLimit", 300)
        model.optimize()
        answer = np.zeros(shape=(Q.shape[0]))
        if model.status != GRB.TIME_LIMIT:
            all_vars = model.getVars()
            values = model.getAttr("X", all_vars)
            names = model.getAttr("VarName", all_vars)
            for name, val in zip(names, values):
                index = int(name.replace("C", ""))
                answer[index] = val
            answer_object = dimod.SampleSet.from_samples(answer, "BINARY", model.ObjVal)
        else:
            answer_object = dimod.SampleSet.from_samples(answer, "BINARY", np.inf)

        return answer_object

    def test_qubo_gurobi(self):
        number_of_trucks = np.random.randint(1, 5)
        capacity_per_truck = np.random.randint(1, 10)
        total_load_per_truck = np.random.randint(10, 20)
        data = LBData.from_random(number_of_trucks, capacity_per_truck, total_load_per_truck)
        model = LBGurobiQUBOUnaryUniformTruckCapacity(data)
        model.model.setParam("TimeLimit", 300)
        model.solve()
        solution = model.solution()
        evaluation_object = LBEvaluation(solution, data)
        optimal_solution_value = number_of_trucks * total_load_per_truck**2
        self.assertNotEqual(model.model.status, GRB.TIME_LIMIT)
        self.assertTrue(evaluation_object.check_solution())
        self.assertEqual(evaluation_object.get_objective(), optimal_solution_value)

    def test_cqm_gurobi(self):
        number_of_trucks = np.random.randint(1, 5)
        capacity_per_truck = np.random.randint(1, 10)
        total_load_per_truck = np.random.randint(10, 20)
        data = LBData.from_random(number_of_trucks, capacity_per_truck, total_load_per_truck)
        model = LBGurobiCQMUnaryUniformTruckCapacity(data)
        model.model.setParam("MIPGap", 1E-5)
        model.model.setParam("TimeLimit", 300)
        model.solve()
        solution = model.solution()
        evaluation_object = LBEvaluation(solution, data)
        optimal_solution_value = number_of_trucks * total_load_per_truck**2
        self.assertNotEqual(model.model.status, GRB.TIME_LIMIT)
        self.assertTrue(evaluation_object.check_solution())
        self.assertEqual(evaluation_object.get_objective(), optimal_solution_value)

    def test_qubo_dwave(self):
        number_of_trucks = np.random.randint(1, 5)
        capacity_per_truck = np.random.randint(1, 10)
        total_load_per_truck = np.random.randint(10, 20)
        data = LBData.from_random(number_of_trucks, capacity_per_truck, total_load_per_truck)
        # Create gurobi model
        model = LBDWAVEQUBOUnaryUniformTruckCapacity(data)
        # Solve problem
        solve_func = TestLBModels.solve_function
        model.solve(solve_func, num_reads=100)

        solution = model.solution()
        evaluation_object = LBEvaluation(solution, data)
        optimal_solution_value = number_of_trucks * total_load_per_truck**2
        self.assertNotEqual(solution["energy"], np.inf)
        self.assertTrue(evaluation_object.check_solution())
        self.assertEqual(evaluation_object.get_objective(), optimal_solution_value)

    def test_cqm_dwave(self):
        number_of_trucks = np.random.randint(1, 5)
        capacity_per_truck = np.random.randint(1, 10)
        total_load_per_truck = np.random.randint(10, 20)
        data = LBData.from_random(number_of_trucks, capacity_per_truck, total_load_per_truck)
        # Create gurobi model
        model = LBDWAVECQMUnaryUniformTruckCapacity(data)
        # Solve problem
        solve_func = TestLBModels.solve_function
        model.solve(solve_func, num_reads=100)

        solution = model.solution()
        self.assertNotEqual(solution["energy"], np.inf)
        evaluation_object = LBEvaluation(solution, data)
        optimal_solution_value = number_of_trucks * total_load_per_truck**2
        self.assertTrue(evaluation_object.check_solution())
        self.assertEqual(evaluation_object.get_objective(), optimal_solution_value)


if __name__ == '__main__':
    unittest.main()
