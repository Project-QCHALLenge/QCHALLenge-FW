import numpy as np
from lb.data.lb_data import LBData
import pandas as pd
import bisect


class LBEvaluation:
    def __init__(self, solution, data : LBData):
        self.data = data
        self.solution = solution
        self.solution["solution"] = LBEvaluation._solution_to_indices(solution["solution"])

    @staticmethod
    def _solution_to_indices(solution : dict) -> list[tuple]:
        converted_solution = []
        for key, value in solution.items():
            if value == 1:
                indices_as_string = tuple(key.split("_")[1:])
                indices = tuple(int(index) for index in indices_as_string)
                converted_solution.append(indices)
        return converted_solution

    def get_objective(self):
        # Objective is Sum L_i**2, where L_i is the total load in the ith truck.
        solution_data = np.array(self.solution["solution"])
        # Group solution by 0 entry of index tuple.
        solution_grouped_by_truck = np.split(solution_data, np.unique(solution_data[:, 0], return_index=1)[1][1:],
                                                  axis=0)
        bisect_wrapper = lambda x: bisect.bisect(self.data.cumulative_quantities, x)
        load_on_trucks = []

        for truck_data in solution_grouped_by_truck:
            # Maps units to product types.
            product_types_on_truck = list(map(bisect_wrapper, truck_data[:, 1]))
            # Counts the number of product of each type on the current truck.
            product_type, count = np.unique(product_types_on_truck, return_counts=True)
            # Calculate L_i, namely number of product type * weight for all product types
            total_load = sum([self.data.weights[product_type[i]] * count[i] for i in range(len(product_type))])
            load_on_trucks.append(total_load)
        # Sum L_i**2
        return sum([l**2 for l in load_on_trucks])



    def _check_every_unit_assigned(self) -> bool:
        solution_data = np.array(self.solution["solution"])
        units, counts = np.unique(solution_data[:, 1], return_counts=True)
        # First condition checks if last entry in unique unit list matches its own index.
        # Correct: L = [0, 1, 2, 3, 4] so L[-1] = len(L)-1.
        # However this will not detect if units were assigned twice or the last unit might not have been assigned
        # If we have 5 units, L = [0,1,2,3] and L = [0,1,2,2,4] must be ruled out.
        # The former case is ruled out by the second statement and the latter case is checked by the third condition.
        every_unit_assigned = (units[-1] == len(units) - 1
                               and units[-1] == self.data.number_of_units -1
                               and max(counts) == 1)

        return bool(every_unit_assigned)

    def _check_every_truck_full(self) -> bool:
        # Get list of tuples from solution structure.
        solution_data = np.array(self.solution["solution"])
        # Group tuples by truck i.e. index 0.
        solution_grouped_by_truck = np.split(solution_data, np.unique(solution_data[:, 0], return_index=1)[1][1:],
                                             axis=0)

        # Checks if the number of assigned units matches the capacity per truck and that each truck has assigned units.
        every_truck_full = ([len(t) for t in solution_grouped_by_truck]
                            == self.data.number_of_trucks * [self.data.capacity_per_truck])

        return every_truck_full

    def check_solution(self) -> list[bool]:
        every_truck_full = self._check_every_truck_full()
        every_unit_assigned = self._check_every_unit_assigned()
        return [every_truck_full, every_unit_assigned]


    def interpret_solution(self) -> pd.DataFrame:
        # The solution format is as follows [(i,j), ...] <=> x_{i,j} = 1.
        # This is not too useful, so this will be interpreted.
        # Solution will be represented as follows:
        # For each truck, the number of products of each type, the total weight resulting from this,
        # the total load and the trucks capacity will be collected.
        records = []
        # Get list of tuples from solution structure.
        solution_data = np.array(self.solution["solution"])
        # Group tuples by truck i.e. index 0.
        solution_grouped_by_truck = np.split(solution_data, np.unique(solution_data[:,0], return_index=1)[1][1:], axis=0)
        # Needed to convert unit number back to product type.
        # cumulative_quantities is the cumsum of the quantities for each product type.
        # In this cumsum the insert index corresponds to product type.
        bisect_wrapper = lambda x: bisect.bisect(self.data.cumulative_quantities, x)

        for truck_data in solution_grouped_by_truck:
            # Truck number is always in the first entry of a tuple.
            truck = truck_data[0, 0]

            # Maps units to product types.
            product_types_on_truck = list(map(bisect_wrapper, truck_data[:, 1]))
            # Counts the frequency of each product type per truck.
            product_type, count = np.unique(product_types_on_truck, return_counts=True)
            # Calculates load per truck, resulting from the assigned units.
            load_per_type = [self.data.weights[t] * c for t, c in zip(product_type, count)]
            # Creates labels for dataframe columns.
            product_type_with_labels = [f"NumberOfUnitsProduct_{i}" for i in product_type]
            product_load_with_labels = [f"LoadOfProduct_{i}" for i in product_type]
            # Defines record to be a dict with "columns" as defined above and count per product type and resulting load.
            record = dict(list(zip(product_type_with_labels, count)) + list(zip(product_load_with_labels, load_per_type)))
            # Adds rest of data.
            record["Truck"] = truck
            record["Capacity"] = self.data.capacity_per_truck
            total_load = sum([self.data.weights[product_type[i]] * count[i] for i in range(len(product_type))])
            record["TotalLoad"] = total_load
            records.append(record)

        # List of all labels the dataframe should have in the end.
        product_type_labels = [f"NumberOfUnitsProduct_{i}" for i in self.data.product_types]
        product_load_labels = [f"LoadOfProduct_{i}" for i in self.data.product_types]
        # To define an order of the columns in the DF, one needs to provide a list of columns in the exact order.
        # Here, a alternating order is preferred:
        # NumberOfUnitsProduct_0, LoadOfProduct_0, NumberOfUnitsProduct_1, LoadOfProduct_1 ....
        # This is done via slicing, hence a list of matching size needs to be created.
        # NumberOfUnitsProduct_0, LoadOfProduct_0 results in len(product_type_labels) * 2 entries.
        # Truck, Capacity, TotalLoad add another 3 entries.
        order_of_columns = [None] * (len(product_type_labels) * 2 + 3)
        # First three columns are Truck, Capacity and TotalLoad.
        order_of_columns[0:3] =[ "Truck", "Capacity", "TotalLoad"]
        # Alternating order.
        order_of_columns[3::2] = product_type_labels
        order_of_columns[4::2] = product_load_labels
        # Create DF from dicts and list of column names in preferred order.
        solution_with_context = pd.DataFrame.from_records(records, columns=order_of_columns).fillna(0)
        return solution_with_context



