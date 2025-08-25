import warnings
from lb.models.lb_abstract_problem import LBAbstractProblem
from lb.data.lb_data import LBData
import numpy as np
import itertools
import bisect

warnings.simplefilter(action='ignore', category=FutureWarning)


class LBQUBOUnaryUniformTruckCapacity(LBAbstractProblem):

    def _assemble_objective_function(self):
        local_c1_matrix = self.c1_full_trucks()
        local_c2_matrix = self.c2_every_unit_is_assigned()
        local_c3_matrix = self.c3_load_on_trucks()
        sqrt_penalty = np.sqrt(self.penalty_parameter)
        local_qubo_matrix = ((1 /sqrt_penalty) * local_c3_matrix + sqrt_penalty
                             *local_c1_matrix)
        # Define Q to be a rank 4 Tensor
        # this makes distributing local constraint matrices to global constraint matrix easy
        self.Q = np.zeros((self.data.number_of_trucks, self.number_of_units,
                           self.data.number_of_trucks, self.number_of_units))

        # Q_{ijil} corresponds to the i-th TxT block matrix on the main diagonal on Q
        self.Q[range(self.data.number_of_trucks), :, range(self.data.number_of_trucks), :] += local_qubo_matrix
        # Q_{ijkj} corresponds to all entries affected by c2 constraint
        self.Q[:, range(self.number_of_units), :, range(self.number_of_units)] += sqrt_penalty * (
            local_c2_matrix)

        # Reshape Q to be a #Units*#Trucks x #Units*#Trucks Matrix
        self.Q = self.Q.reshape(self.number_of_units * self.data.number_of_trucks,
                                self.number_of_units * self.data.number_of_trucks)
        self.A = sqrt_penalty * (self.data.capacity_per_truck**2 * self.data.number_of_trucks + self.data.number_of_units
                                  )
    def _assemble_constraints(self):
        pass

    def c1_full_trucks(self):
        local_qubo_matrix = np.diag(self.number_of_units * [1-2*self.data.capacity_per_truck])
        upper_triangular_indices = np.triu_indices(self.number_of_units, 1)
        local_qubo_matrix[upper_triangular_indices] = 2
        return local_qubo_matrix

    def c2_every_unit_is_assigned(self):
        local_qubo_matrix = np.diag(self.data.number_of_trucks * [-1])
        upper_triangular_indices = np.triu_indices(self.data.number_of_trucks, 1)
        local_qubo_matrix[upper_triangular_indices] = 2
        return local_qubo_matrix

    def c3_load_on_trucks(self):
        weights = [self.weights[i]  for i in range(len(self.quantities))  for j in range(self.quantities[i])]
        squared_weights = list(map(lambda x: x**2 , weights))
        local_qubo_matrix = np.diag(squared_weights)
        upper_triangular_indices = np.triu_indices(self.number_of_units, 1)
        for row_index, column_index in zip(*upper_triangular_indices):
            local_qubo_matrix[row_index, column_index] = weights[column_index] * weights[row_index] * 2
        return local_qubo_matrix

    def _create_data_structures(self):
        self.quantities = [product["quantity"] for product in self.data.products]
        self.number_of_units = sum(self.quantities)
        self.weights = [product["weight"] for product in self.data.products]

        weights_and_quantities = zip(self.weights, self.quantities)
        weights_and_quantities_sorted = np.array(sorted(weights_and_quantities, key=lambda x: -x[0]))

        sorted_quantities_cum_sum = np.cumsum(weights_and_quantities_sorted[:, 1])

        index_top_c_heaviest = bisect.bisect(sorted_quantities_cum_sum, self.data.capacity_per_truck)
        top_c_heaviest_weights = weights_and_quantities_sorted[:index_top_c_heaviest, 0]
        top_c_heaviest_quantities = weights_and_quantities_sorted[:index_top_c_heaviest, 1]

        self.avg_load_per_truck = np.dot(self.weights, self.quantities)/ self.data.number_of_trucks


        self.penalty_parameter = (2 * max(self.weights) * self.avg_load_per_truck) + 1
    def __init__(self, data: LBData):
        self.Q = None
        self.penalty_parameter = 0
        self.number_of_units = None
        super().__init__(data)




