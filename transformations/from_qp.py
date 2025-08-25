import os

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_docplex_mp, to_gurobipy
from qiskit_optimization.converters import QuadraticProgramToQubo

class FromQP:

    def __init__(self, qp: QuadraticProgram):
        self.__qp_model = qp

    def to_cplex(self):
        cplex = to_docplex_mp(self.__qp_model)
        return cplex
    
    def to_gurobi(self):
        gurobi = to_gurobipy(self.__qp_model)
        return gurobi
    
    def to_lp(self, file_path):
        self.__qp_model.write_to_lp_file(os.path.join(os.path.abspath(""), f'{file_path}.lp'))
        return file_path
    
    def to_qubo_qiskit(self):
        qubo_qiskit = QuadraticProgramToQubo().convert(self.__qp_model)
        return qubo_qiskit
