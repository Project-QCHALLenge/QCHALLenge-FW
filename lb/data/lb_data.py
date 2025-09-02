import json
import numpy as np
from abstract.data.abstract_data import AbstractData
import math


class LBData(AbstractData):
    def __init__(self, params: dict):
        self.number_of_trucks = params["number_of_trucks"]
        self.products = params["products"]
        self.quantities = [product["quantity"] for product in self.products]

        assert(all(quantity >= 1 for quantity in self.quantities))

        self.number_of_units = sum(self.quantities)

        self.capacity_per_truck = self.number_of_units / self.number_of_trucks

        self.product_types = range(len(self.quantities))

        # defines a map m that maps product type to indices
        # Quantity: 2, Weight 3, #Trucks 1 -> x_11, x_12 and w_1, w_2 where w_1 = w_2.
        # Hence, the number of weight multiplications can be reduced
        self.cumulative_quantities = np.cumsum(self.quantities)
        self.product_type_to_indices = ([range(self.cumulative_quantities[0])]
                                        + [range(self.cumulative_quantities[i], self.cumulative_quantities[i + 1])
                                           for i in range(len(self.cumulative_quantities) - 1)])

        self.weights = [product["weight"] for product in self.products]

        assert (all(weight >= 1 for weight in self.weights))
        assert (int(self.capacity_per_truck) == self.capacity_per_truck)

    @classmethod
    def create_problem(cls, number_of_trucks: int, products: list[tuple]):
        products_dict = [{"quantity": p[0], "weight": p[1]} for p in products]
        params = {"number_of_trucks": number_of_trucks, "products": products_dict}
        return cls(params)

    @classmethod
    def from_json(cls, path_to_file: str):
        with open(path_to_file, "r") as file:
            data_from_file = json.load(file)

        return cls(data_from_file)

    @classmethod
    def from_random(cls, number_of_trucks, capacity_per_truck, total_load_per_truck):
        assert number_of_trucks > 0 and type(number_of_trucks) == int
        assert capacity_per_truck > 0 and type(capacity_per_truck) == int
        assert total_load_per_truck > 0 and type(total_load_per_truck) == int
        assert total_load_per_truck >= capacity_per_truck
        # Generates partially random instance with total_load_per_truck as optimal objective value
        total_weights = np.zeros(shape=(number_of_trucks, capacity_per_truck))
        for truck in range(number_of_trucks):
            weights = [math.floor(total_load_per_truck / capacity_per_truck) for _ in range(capacity_per_truck)]
            i = 0
            while sum(weights) < total_load_per_truck:
                weights[i] += 1
                i += 1
            minimal_weight_before_perturbation = min(weights)
            for i in range(math.floor(capacity_per_truck / 2)):
                random_perturbation = np.random.randint(0, minimal_weight_before_perturbation)
                weights[i] += random_perturbation
                weights[-i] -= random_perturbation
            total_weights[truck] = weights

        unique_weights, counts = np.unique(total_weights, return_counts=True)
        product_list = [i for i in zip(counts, unique_weights)]
        return cls.create_problem(number_of_trucks, product_list)





