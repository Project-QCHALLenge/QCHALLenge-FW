import os
import dimod
import gurobipy as gp
from docplex.mp.model_reader import ModelReader
from qiskit_optimization.problems import QuadraticProgram

from transformations.from_bqm import FromBQM
from transformations.from_cqm import FromCQM
from transformations.from_qp import FromQP

class FromLP:

    def __init__(self, file_path):
        self.__file_path = file_path

    def to_bqm(self):
        cqm = self.to_cqm()
        bqm, conv = FromCQM(cqm).to_bqm()
        return bqm, conv

    def to_cqm(self):
        with open(f'{self.__file_path}.lp', 'rb') as f:
            cqm = dimod.lp.load(f)
        return cqm
    
    def to_cplex(self):
        model = ModelReader.read(f'{self.__file_path}.lp', ignore_names=True)
        return model
    
    def to_qp(self):
        qp = QuadraticProgram()
        qp.read_from_lp_file(os.path.join(os.path.abspath(""), f'{self.__file_path}.lp'))
        return qp
    
    def to_qubo_qiskit(self):
        qp = self.to_qp()
        qubo_qiskit = FromQP(qp).to_qubo_qiskit()
        return qubo_qiskit
    
    def to_gurobi(self):
        gurobi_model = gp.Model().read(f'{self.__file_path}.lp')
        return gurobi_model
    
    def to_matrix(self):
        cqm = self.to_cqm()
        matrix, conv = FromCQM(cqm).to_matrix()
        return matrix, conv
    
    def to_matrix_with_dict(self):
        bqm, _ = self.to_bqm()
        matrix, conv = FromBQM(bqm).to_matrix()
        return matrix, conv
