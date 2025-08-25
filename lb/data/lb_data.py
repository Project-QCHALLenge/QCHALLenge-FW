import json
import numpy as np


class LBData:
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

        assert(all(weight >= 1 for weight in self.weights))
        assert(int(self.capacity_per_truck) == self.capacity_per_truck)


    @classmethod
    def create_problem(cls, number_of_trucks: int, products: list[tuple]):
        products_dict = [{"quantity": p[0], "weight": p[1]} for p in products]
        params = {"number_of_trucks": number_of_trucks, "products": products_dict}
        return cls(params)

    @classmethod
    def create_problem_from_file(cls, path_to_file: str):
        with open(path_to_file, "r") as file:
            data_from_file = json.load(file)

        return cls(data_from_file)





