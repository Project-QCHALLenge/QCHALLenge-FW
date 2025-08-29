from unittest import TestCase
import unittest.mock
from lb.data.lb_data import LBData
import numpy as np
import json


class TestLBDataCreateProblem(TestCase):
    number_of_trucks = 4
    products = [(5, 1), (4, 76), (9, 23), (2, 24)]
    data_object = LBData.create_problem(number_of_trucks, products)

    mock_file_dict = {"number_of_trucks": 4, "products":
        [{"quantity": quant, "weight": weight} for quant, weight in products]}
    mock_file_content = json.dumps(mock_file_dict)

    def test_create_problem_from_file_equality(self):
        with unittest.mock.patch(
            "builtins.open",
            new = unittest.mock.mock_open(read_data=self.mock_file_content),
            create=True
        ) as file_mock:
            data_object_from_file = LBData.from_json("")
            equality_statement = ((data_object_from_file.number_of_trucks  == self.data_object.number_of_trucks)
                                  and (data_object_from_file.products == self.data_object.products))
            self.assertTrue(equality_statement)


    def test_create_problem_negative_quantities(self):
        number_of_trucks = 4
        products = [(-1, 1), (10, 76), (9, 23), (2, 24)]
        self.assertRaises(AssertionError, lambda: LBData.create_problem(number_of_trucks, products))

    def test_create_problem_negative_weight(self):
        number_of_trucks = 4
        products = [(1, 1), (8, 76), (9, -23), (2, 24)]
        self.assertRaises(AssertionError, lambda: LBData.create_problem(number_of_trucks, products))


    def test_create_problem_number_of_trucks(self):
        self.assertEqual(self.data_object.number_of_trucks, 4)

    def test_create_problem_number_of_units(self):
        self.assertEqual(self.data_object.number_of_units, 20)

    def test_create_problem_weights(self):
        self.assertEqual(self.data_object.weights, [1, 76, 23, 24])

    def test_create_problem_quantities(self):
        self.assertEqual(self.data_object.quantities, [5, 4, 9, 2])

    def test_create_problem_product_types(self):
        self.assertEqual(self.data_object.product_types, range(4))

    def test_create_problem_product_type_to_indices(self):
        self.assertEqual(self.data_object.product_type_to_indices,
                         [range(5), range(5, 9), range(9, 18), range(18, 20)])

    def test_create_problem_cumulative_quantities(self):
        list_of_differences = [actual_value - expected_value
                               for actual_value, expected_value
                               in zip(self.data_object.cumulative_quantities, [5, 9, 18, 20])]
        self.assertEqual(sum(list_of_differences), 0)

    def test_create_problem_capacity_per_truck(self):
        self.assertEqual(self.data_object.capacity_per_truck, 5)




