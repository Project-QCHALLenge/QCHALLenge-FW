from lb.models.lb_abstract_problem import LBAbstractProblem
from lb.data.lb_data import LBData
import numpy as np


class LBCQMUniformTruckCapacity(LBAbstractProblem):

    def c1_each_truck_is_full(self):
        local_A = np.ones(self.number_of_units)
        local_rhs = np.zeros(self.data.number_of_trucks) + self.data.capacity_per_truck
        return local_A, local_rhs

    def c2_each_unit_is_assigned(self):
        local_A = np.ones(self.data.number_of_trucks)
        local_rhs = np.zeros(self.number_of_units) + 1
        return local_A, local_rhs

    def obj_load_on_trucks(self):
        weights = [self.weights[i]  for i in range(len(self.quantities))  for j in range(self.quantities[i])]
        squared_weights = list(map(lambda x: x**2, weights))
        objective_matrix = np.diag(squared_weights)
        upper_triangular_indices = np.triu_indices(self.number_of_units, 1)
        for row_index, column_index in zip(*upper_triangular_indices):
            objective_matrix[row_index, column_index] = weights[column_index] * weights[row_index] * 2
        return objective_matrix

    def _assemble_objective_function(self):
        local_objective_matrix = self.obj_load_on_trucks()

        # Q_{ijil} corresponds to the i-th TxT block matrix on the main diagonal on Q
        self.Q[range(self.data.number_of_trucks), :, range(self.data.number_of_trucks), :] += local_objective_matrix

        # Reshape Q to be a #Units*#Trucks x #Units*#Trucks Matrix
        self.Q = self.Q.reshape(self.number_of_units * self.data.number_of_trucks,
                                self.number_of_units * self.data.number_of_trucks)

    def _assemble_constraints(self):
        each_truck_is_full_matrix, each_truck_is_full_rhs = self.c1_each_truck_is_full()
        each_unit_is_assigned_matrix, each_unit_assigned_rhs = self.c2_each_unit_is_assigned()

        lower_block_matrix_range = range(self.data.number_of_trucks, self.data.number_of_trucks + self.number_of_units)

        self.A[range(self.data.number_of_trucks), range(self.data.number_of_trucks), :] = each_truck_is_full_matrix
        self.A[lower_block_matrix_range, :, range(self.number_of_units)] = each_unit_is_assigned_matrix

        # Flattens constraint tensor to constraint matrix
        self.A = self.A.reshape((self.data.number_of_trucks + self.number_of_units,
                           self.data.number_of_trucks * self.number_of_units))

        # Stacks both local constraint rhs to one global one, column vector of size UxT
        self.rhs = np.hstack((each_truck_is_full_rhs, each_unit_assigned_rhs))

    def _create_data_structures(self):
        pass

    def __init__(self, data: LBData):
        super().__init__(data)
        self.quantities = self.data.quantities
        self.number_of_units = self.data.number_of_units
        self.weights = self.data.weights
        # Define Q to be a rank 4 Tensor
        # this makes distribution local objective matrices to global objective matrix easy
        self.Q = np.zeros((self.data.number_of_trucks, self.number_of_units,
                           self.data.number_of_trucks, self.number_of_units))

        # Define A to be a rank 3 Tensor,
        # this makes distribution of local constraint matrices to global constraint matrix easy
        self.A = np.zeros((self.data.number_of_trucks + self.number_of_units,
                           self.data.number_of_trucks, self.number_of_units))

