from dimod import BinaryQuadraticModel

class FromBQM:

    def __init__(self, bqm: BinaryQuadraticModel):
        self.__bqm_model = bqm

    def to_matrix(self):
        num_variables = len(self.__bqm_model.variables)
        qubo_matrix = [[0] * num_variables for _ in range(num_variables)]

        variable_indices = {var: idx for idx, var in enumerate(self.__bqm_model.variables)}

        for variable, bias in self.__bqm_model.linear.items():
            idx = variable_indices[variable]
            qubo_matrix[idx][idx] = bias

        for (variable1, variable2), bias in self.__bqm_model.quadratic.items():
            idx1 = variable_indices[variable1]
            idx2 = variable_indices[variable2]
            qubo_matrix[idx1][idx2] = bias / 2
            qubo_matrix[idx2][idx1] = bias / 2

        return qubo_matrix, variable_indices