import copy
from unittest import TestCase
from unittest.mock import patch

from lb.evaluation.evaluation import LBEvaluation
from unittest import mock


class TestLBEvaluation(TestCase):
    # Product_1 = [0, 1,2]
    # Product_2 = [3,4]
    # Product_3 = [5, 6,7,8,9]
    solution = {"solution": {'x_0_0': 1,'x_0_1': 1, 'x_0_3': 1, 'x_0_5': 1, 'x_1_2': 1,
                'x_1_6': 1, 'x_2_4': 1, 'x_2_7': 1, 'x_3_8': 1, 'x_3_9': 1}, "energy": 1}
    mock_data = mock.Mock()
    mock_data.cumulative_quantities = [3, 5, 10]
    mock_data.weights = [10, 20, 30]
    mock_data.capacity_per_truck = 4
    mock_data.number_of_trucks = 4
    mock_data.product_types = range(3)
    mock_data.number_of_units = 10
    mock_return_value = [(0, 0), (0, 1), (0, 3), (0, 5), (1, 2), (1, 6), (2, 4), (2, 7), (3, 8), (3, 9)]
    with patch("lb.evaluation.evaluation.LBEvaluation._solution_to_indices", return_value = mock_return_value):
        interpreted_solution = LBEvaluation(copy.deepcopy(copy.deepcopy(solution)), data=mock_data).interpret_solution()

    def test_interpret_solution_capacity_per_truck(self):
        self.assertEqual(list(self.interpreted_solution["Capacity"]), 4 * [self.mock_data.capacity_per_truck])

    def test_interpret_solution_unit_assignments(self):
        assignment_correct_first_product = list(self.interpreted_solution["NumberOfUnitsProduct_0"]) == [2, 1, 0, 0]
        assignment_correct_second_product = list(self.interpreted_solution["NumberOfUnitsProduct_1"]) == [1, 0, 1, 0]
        assignment_correct_third_product = list(self.interpreted_solution["NumberOfUnitsProduct_2"]) == [1, 1, 1, 2]
        self.assertTrue(assignment_correct_second_product
                        and assignment_correct_third_product
                        and assignment_correct_first_product)

    def test_interpret_solution_loads_on_trucks(self):
        load_correct_first_product = list(self.interpreted_solution["LoadOfProduct_0"]) == [20, 10, 0, 0]
        load_correct_second_product = list(self.interpreted_solution["LoadOfProduct_1"]) == [20, 0, 20, 0]
        load_correct_third_product = list(self.interpreted_solution["LoadOfProduct_2"]) == [30, 30, 30, 60]
        self.assertTrue(load_correct_third_product
                        and load_correct_second_product
                        and load_correct_first_product)

    def test_interpret_solution_total_load_per_truck(self):
        self.assertTrue(list(self.interpreted_solution["TotalLoad"]) == [70, 40, 50, 60])

    def test__check_every_unit_assigned(self):
        solution = copy.deepcopy(self.solution)
        assigned = LBEvaluation(solution, data=self.mock_data)._check_every_unit_assigned()
        self.assertTrue(assigned)

    @patch("lb.evaluation.evaluation.LBEvaluation._solution_to_indices")
    def test__check_every_unit_assigned_fail_missing_unit(self, mock__solution_to_indices):
        mock__solution_to_indices.return_value =   \
            [(0, 1), (0, 3), (0, 5), (1, 2), (1, 6), (2, 4), (2, 7), (3, 8), (3, 9)]
        assigned = LBEvaluation({"solution" : [], "energy": 1}, data=self.mock_data)._check_every_unit_assigned()
        self.assertFalse(assigned)

    @patch("lb.evaluation.evaluation.LBEvaluation._solution_to_indices")
    def test__check_every_unit_assigned_fail_duplicate_unit(self, mock__solution_to_indices):
        mock__solution_to_indices.return_value = \
            [(0, 1), (0, 1), (0, 3), (0, 5), (1, 2), (1, 6), (2, 4), (2, 7), (3, 8), (3, 9)]
        assigned = LBEvaluation({"solution" : [], "energy": 1}, data=self.mock_data)._check_every_unit_assigned()
        self.assertFalse(assigned)

    @patch("lb.evaluation.evaluation.LBEvaluation._solution_to_indices")
    def test__check_every_unit_assigned_fail_last_unit_missing(self, mock__solution_to_indices):
        mock__solution_to_indices.return_value = \
            [(0, 0), (0, 1), (0, 3), (0, 5), (1, 2), (1, 6), (2, 4), (2, 7), (3, 8)]
        assigned = LBEvaluation({"solution" : [], "energy": 1}, data=self.mock_data)._check_every_unit_assigned()
        self.assertFalse(assigned)

    def test__check_every_truck_full_fail_capacity_not_met(self):
        solution = copy.deepcopy(self.solution)
        assigned = LBEvaluation(solution, data=self.mock_data)._check_every_truck_full()
        self.assertFalse(assigned)

    @patch("lb.evaluation.evaluation.LBEvaluation._solution_to_indices")
    def test__check_every_truck_full_fail_truck_left_out(self, mock__solution_to_indices):
        mock__solution_to_indices.return_value = [(0, 0), (0, 1), (0, 3), (0, 5), (1, 2), (1, 6), (1, 4), (1, 7)]
        assigned = LBEvaluation({"solution" : [], "energy": 1}, data=self.mock_data)._check_every_truck_full()
        self.assertFalse(assigned)

    @patch("lb.evaluation.evaluation.LBEvaluation._solution_to_indices")
    def test__check_every_truck_full(self, mock__solution_to_indices):
        mock__solution_to_indices.return_value = [(0, 0), (0, 1), (0, 3), (0, 5), (1, 2), (1, 6), (1, 10), (1, 7),
                                 (2, 8), (2, 9), (2, 11), (2, 14), (3, 4), (3, 15), (3, 11), (3, 12)]
        assigned = LBEvaluation({"solution" : [], "energy": 1}, data=self.mock_data)._check_every_truck_full()
        self.assertTrue(assigned)

    def test__solution_to_indices(self):
        solution = {"solution": {'x_0_0': 1,'x_0_1': 1, 'x_0_3': 1, 'x_0_5': 1, 'x_1_2': 1,
                'x_1_6': 1, 'x_2_4': 1, 'x_2_7': 1, 'x_3_8': 1, 'x_3_9': 1}, "energy": 1}
        expected_return_value = [(0, 0), (0, 1), (0, 3), (0, 5), (1, 2), (1, 6), (2, 4), (2, 7), (3, 8), (3, 9)]
        non_zero_indices = LBEvaluation._solution_to_indices(solution)
        self.assertEqual(self.mock_return_value, expected_return_value)

