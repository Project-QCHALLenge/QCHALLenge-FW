import unittest
from tl import TLScip, TLEvaluation
from tl import TLData
from tl.utils.tl_data_generic import TruckParameters
import pandas as pd
import numpy as np
from tl import TLPlot
from gurobipy import GRB



class MyTestCase(unittest.TestCase):
    def test_trivial_case(self):
        boxes = [{"index": i, "length": 1, "width": 1, "height": 0, "weight": 1} for i in range(18)]
        truck_parameters = TruckParameters(6, 3, 0, 18)
        data = TLData(truck_parameters, pd.DataFrame(boxes))
        model = TLScip(data)
        model.model.setParam('limits/time', 300)
        answer = model.solve(**{"TimeLimit": 300})
        evaluation = TLEvaluation(data=data, solution=answer)

        self.assertNotEqual(model.model.getStatus(), "timelimit")
        self.assertEqual(evaluation.get_objective(), 18)  # add assertion here

    def test_random(self):
        truck_length = np.random.randint(2, 20)
        truck_width = np.random.randint(1, truck_length)
        number_of_boxes = np.random.randint(1, 20)
        box_length = truck_length / number_of_boxes
        box_width = truck_width
        truck_capacity = np.random.randint(1, 20)
        box_weights = np.random.random(size=(number_of_boxes))
        box_weights_rescaled = (truck_capacity) * (box_weights / sum(box_weights))
        boxes = [{"index": i, "length": box_length, "width": box_width, "height": 0, "weight": box_weights_rescaled[i]} for i in range(number_of_boxes)]
        truck_parameters = TruckParameters(truck_length, truck_width, 0, truck_capacity)
        data = TLData(truck_parameters, pd.DataFrame(boxes))
        model = TLScip(data)
        model.model.setParam('limits/time', 300)
        answer = model.solve(**{"TimeLimit": 300})
        evaluation = TLEvaluation(data=data, solution=answer)
        nr_of_violated_constraint = 0
        for constraint, violations in evaluation.check_solution().items():
            if len(violations) > 0:
                nr_of_violated_constraint += 1

        self.assertNotEqual(model.model.getStatus(), "timelimit")
        self.assertEqual(nr_of_violated_constraint, 0)
        self.assertLessEqual(abs(evaluation.get_objective() - truck_length*truck_width), 1)  # add assertion here

    def test_rounding_issue(self):
        truck_length = 8
        truck_width = 3
        number_of_boxes = 11
        box_length = truck_length / number_of_boxes
        box_width = truck_width
        truck_capacity = 2
        box_weights = np.random.random(size=(number_of_boxes))
        box_weights_rescaled = (truck_capacity) * (box_weights / sum(box_weights))
        boxes = [{"index": i, "length": box_length, "width": box_width, "height": 0, "weight": box_weights_rescaled[i]
                  } for i in range(number_of_boxes)]
        truck_parameters = TruckParameters(truck_length, truck_width, 0, truck_capacity)
        data = TLData(truck_parameters, pd.DataFrame(boxes))
        model = TLScip(data)
        model.model.setParam('limits/time', 300)
        answer = model.solve(**{"TimeLimit": 300})
        evaluation = TLEvaluation(data=data, solution=answer)
        self.assertNotEqual(model.model.getStatus(), "timelimit")
        self.assertEqual(evaluation.get_objective(), truck_length * truck_width)  # add assertion here

    def test_width_length_mix_up_2(self):
        truck_length = 13
        truck_width = 14
        number_of_boxes = 13
        box_length = truck_length / number_of_boxes
        box_width = truck_width
        truck_capacity = np.random.randint(1, 20)
        box_weights = np.zeros(shape=(number_of_boxes))
        boxes = [{"index": i, "length": box_length, "width": box_width, "height": 0, "weight": box_weights[i]} for i in range(number_of_boxes)]
        truck_parameters = TruckParameters(truck_length, truck_width, 0, truck_capacity)
        data = TLData(truck_parameters, pd.DataFrame(boxes))
        model = TLScip(data)
        model.model.setParam('limits/time', 300)
        model.model.optimize()
        self.assertNotEqual(model.model.getStatus(), "timelimit")
        self.assertNotEqual(model.model.getStatus(), "infeasible")

    def test_width_length_mix_up_1(self):
        truck_length = 14
        truck_width = 13
        number_of_boxes = 14
        box_length = truck_length / number_of_boxes
        box_width = truck_width
        truck_capacity = np.random.randint(1, 20)
        box_weights = np.zeros(shape=(number_of_boxes))
        boxes = [{"index": i, "length": box_length, "width": box_width, "height": 0, "weight": box_weights[i]
                 } for i in range(number_of_boxes)]
        truck_parameters = TruckParameters(truck_length, truck_width, 0, truck_capacity)
        data = TLData(truck_parameters, pd.DataFrame(boxes))
        model = TLScip(data)
        model.model.setParam('limits/time', 300)
        model.model.optimize()
        self.assertNotEqual(model.model.getStatus(), "timelimit")
        self.assertNotEqual(model.model.getStatus(), "infeasible")

    def test_pick_larger_box(self):
        boxes = [{"index": 0, "length": 4, "width": 3, "height": 0, "weight": 12},
                 {"index": 1, "length": 3, "width": 3, "height": 0, "weight": 9}]
        truck_parameters = TruckParameters(6, 3, 0, 18)
        data = TLData(truck_parameters, pd.DataFrame(boxes))

        model = TLScip(data)
        model.model.setParam('limits/time', 300)
        answer = model.solve(**{"TimeLimit": 300})
        evaluation = TLEvaluation(data=data, solution=answer)

        self.assertEqual(evaluation.get_objective(), 12)  # add assertion here

    def test_one_box_too_heavy(self):
        boxes = [{"index": 0, "length": 1, "width": 1, "height": 0, "weight": 20}]
        truck_parameters = TruckParameters(6, 3, 0, 18)
        data = TLData(truck_parameters, pd.DataFrame(boxes))

        model = TLScip(data)
        model.model.setParam('limits/time', 300)
        answer = model.solve(**{"TimeLimit": 300})
        evaluation = TLEvaluation(data=data, solution=answer)

        self.assertNotEqual(model.model.getStatus(), "timelimit")
        self.assertEqual(evaluation.get_objective(), 0)  # add assertion here

    def test_one_box_too_long(self):
        boxes = [{"index": 0, "length": 7, "width": 1, "height": 0, "weight": 1}]
        truck_parameters = TruckParameters(6, 3, 0, 18)
        data = TLData(truck_parameters, pd.DataFrame(boxes))

        model = TLScip(data)
        model.model.setParam('limits/time', 300)
        answer = model.solve(**{"TimeLimit": 300})

        self.assertNotEqual(model.model.getStatus(), "timelimit")
        self.assertEqual(answer, None)  # add assertion here

    def test_one_box_too_wide(self):
        boxes = [{"index": 0, "length": 1, "width": 7, "height": 0, "weight": 1}]
        truck_parameters = TruckParameters(6, 3, 0, 18)
        data = TLData(truck_parameters, pd.DataFrame(boxes))

        model = TLScip(data)
        model.model.setParam('limits/time', 300)
        answer = model.solve(**{"TimeLimit": 300})

        self.assertNotEqual(model.model.getStatus(), "timelimit")
        self.assertEqual(answer, None)  # add assertion here


if __name__ == '__main__':
    unittest.main()
