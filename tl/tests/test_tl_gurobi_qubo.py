import unittest
from tl import TLQubo, TLEvaluation
from tl import TLData
from tl.utils.tl_data_generic import TruckParameters
import pandas as pd
import numpy as np
from tl import TLPlot
from gurobipy import GRB
import dimod
import gurobipy as gp
from gurobipy import GRB


class TestTLQUBO(unittest.TestCase):

    @staticmethod
    def create_gurobi_model_from_Q(Q):
        model = gp.Model("QUBO")
        variables = model.addVars(range(Q.shape[0]), vtype=GRB.BINARY)
        x = gp.MVar.fromlist(list(variables.values()))
        model.setObjective(x.T @ Q @ x, GRB.MINIMIZE)
        return model

    @staticmethod
    def solve_function(Q, *args, **kwargs):
        if type(Q) == list:
            Q = np.array(Q)
        if type(Q) == dimod.BinaryQuadraticModel:
            Q = Q.to_numpy_matrix(variable_order=Q.variables)
        model = TestTLQUBO.create_gurobi_model_from_Q(Q)
        model.setParam("TimeLimit", 3000)
        model.optimize()
        answer = np.zeros(shape=(Q.shape[0]))
        if not (model.status == GRB.TIME_LIMIT or model.status == GRB.INFEASIBLE):
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

    def test_trivial_case(self):
        boxes = [{"index": i, "length": 1, "width": 1, "height": 0, "weight": 1} for i in range(9)]
        truck_parameters = TruckParameters(3, 3, 0, 9)
        data = TLData(truck_parameters, pd.DataFrame(boxes))

        gurobi_model = TLQubo(data)
        answer = gurobi_model.solve(TestTLQUBO.solve_function, lagrange_multiplier=10000000)
        evaluation = TLEvaluation(data=data, solution=answer)

        plt = TLPlot(evaluation).plot_solution()
        plt.show()

        self.assertNotEqual(answer["energy"], np.inf)
        self.assertEqual(evaluation.get_objective(), 9)  # add assertion here

    def test_random(self):
        truck_length = np.random.randint(2, 20)
        truck_width = np.random.randint(1, truck_length)
        number_of_boxes = np.random.randint(1, 20)
        box_length = truck_length / number_of_boxes
        box_width = truck_width
        truck_capacity = np.random.randint(1, 20)
        box_weights = np.random.random(size=(number_of_boxes))
        box_weights_rescaled = (truck_capacity) * (box_weights / sum(box_weights))
        boxes = [{"index": i, "length": box_length, "width": box_width, "height": 0, "weight": box_weights_rescaled[i]}
                 for i in range(number_of_boxes)]
        truck_parameters = TruckParameters(truck_length, truck_width, 0, truck_capacity)
        data = TLData(truck_parameters, pd.DataFrame(boxes))
        gurobi_model = TLQubo(data)
        answer = gurobi_model.solve(TestTLQUBO.solve_function)
        evaluation = TLEvaluation(data=data, solution=answer)
        nr_of_violated_constraint = 0
        for constraint, violations in evaluation.check_solution().items():
            if len(violations) > 0:
                nr_of_violated_constraint += 1

        self.assertNotEqual(answer["energy"], np.inf)
        self.assertEqual(nr_of_violated_constraint, 0)
        self.assertLessEqual(abs(evaluation.get_objective() - truck_length*truck_width), 1)  # add assertion here

    def test_rounding_issue(self):
        truck_length = 8
        truck_width = 3
        number_of_boxes = 8
        box_length = int(truck_length / number_of_boxes)
        box_width = truck_width
        truck_capacity = 20
        box_weights = np.random.random(size=(number_of_boxes))
        box_weights_rescaled = 1 + 0*(truck_capacity) * (box_weights / sum(box_weights))
        boxes = [{"index": i, "length": box_length, "width": box_width, "height": 0, "weight": box_weights_rescaled[i]}
                 for i in range(number_of_boxes)]
        truck_parameters = TruckParameters(truck_length, truck_width, 0, truck_capacity)
        data = TLData(truck_parameters, pd.DataFrame(boxes))
        gurobi_model = TLQubo(data)
        answer = gurobi_model.solve(TestTLQUBO.solve_function)
        evaluation = TLEvaluation(data=data, solution=answer)
        self.assertNotEqual(gurobi_model.model.status, GRB.TIME_LIMIT)
        self.assertEqual(evaluation.get_objective(), truck_length * truck_width)  # add assertion here

    def test_fail_width_length_mix_up(self):
        truck_length = 13
        truck_width = 14
        number_of_boxes = 16
        box_length = truck_length / number_of_boxes
        box_width = truck_width
        truck_capacity = np.random.randint(1, 20)
        box_weights = np.zeros(shape=(number_of_boxes))
        boxes = [{"index": i, "length": box_length, "width": box_width, "height": 0, "weight": box_weights[i]} for i in range(number_of_boxes)]
        truck_parameters = TruckParameters(truck_length, truck_width, 0, truck_capacity)
        data = TLData(truck_parameters, pd.DataFrame(boxes))
        gurobi_model = TLQubo(data)
        answer = gurobi_model.solve(TestTLQUBO.solve_function)
        self.assertNotEqual(answer["energy"], np.inf)

    def test_width_length_mix_up(self):
        truck_length = 13
        truck_width = 13
        number_of_boxes = 16
        box_length = truck_length / number_of_boxes
        box_width = truck_width
        truck_capacity = np.random.randint(1, 20)
        box_weights = np.zeros(shape=(number_of_boxes))
        boxes = [{"index": i, "length": box_length, "width": box_width, "height": 0, "weight": box_weights[i]} for i in range(number_of_boxes)]
        truck_parameters = TruckParameters(truck_length, truck_width, 0, truck_capacity)
        data = TLData(truck_parameters, pd.DataFrame(boxes))
        gurobi_model = TLQubo(data)
        answer = gurobi_model.solve(TestTLQUBO.solve_function)
        self.assertNotEqual(answer["energy"], np.inf)

    def test_pick_larger_box(self):
        boxes = [{"index": 0, "length": 4, "width": 3, "height": 0, "weight": 12},
                 {"index": 1, "length": 3, "width": 3, "height": 0, "weight": 9}]
        truck_parameters = TruckParameters(6, 3, 0, 18)
        data = TLData(truck_parameters, pd.DataFrame(boxes))

        gurobi_model = TLQubo(data)
        answer = gurobi_model.solve(TestTLQUBO.solve_function)
        evaluation = TLEvaluation(data=data, solution=answer)

        self.assertNotEqual(answer["energy"], np.inf)
        self.assertEqual(evaluation.get_objective(), 12)  # add assertion here

    def test_one_box_too_heavy(self):
        boxes = [{"index": 0, "length": 1, "width": 1, "height": 0, "weight": 20}]
        truck_parameters = TruckParameters(6, 3, 0, 18)
        data = TLData(truck_parameters, pd.DataFrame(boxes))

        gurobi_model = TLQubo(data)
        answer = gurobi_model.solve(TestTLQUBO.solve_function)
        evaluation = TLEvaluation(data=data, solution=answer)

        self.assertNotEqual(answer["energy"], np.inf)
        self.assertEqual(evaluation.get_objective(), 0)  # add assertion here


    def test_one_box_too_long(self):
        boxes = [{"index": 0, "length": 7, "width": 1, "height": 0, "weight": 20}]
        truck_parameters = TruckParameters(6, 3, 0, 18)
        data = TLData(truck_parameters, pd.DataFrame(boxes))

        gurobi_model = TLQubo(data)
        answer = gurobi_model.solve(TestTLQUBO.solve_function)
        evaluation = TLEvaluation(data=data, solution=answer)

        self.assertNotEqual(answer["energy"], np.inf)
        self.assertEqual(evaluation.get_objective(), 0)  # add assertion here

    def test_one_box_too_wide(self):
        boxes = [{"index": 0, "length": 1, "width": 7, "height": 0, "weight": 20}]
        truck_parameters = TruckParameters(6, 3, 0, 18)
        data = TLData(truck_parameters, pd.DataFrame(boxes))

        gurobi_model = TLQubo(data)
        answer = gurobi_model.solve(TestTLQUBO.solve_function)
        evaluation = TLEvaluation(data=data, solution=answer)

        self.assertNotEqual(answer["energy"], np.inf)
        self.assertEqual(evaluation.get_objective(), 0)  # add assertion here


if __name__ == '__main__':
    unittest.main()
