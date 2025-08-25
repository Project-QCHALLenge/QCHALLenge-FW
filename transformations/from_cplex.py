from typing import Union
from pathlib import Path

from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model as CplexModel

from transformations.from_lp import FromLP
from transformations.from_qp import FromQP
from transformations.from_cqm import FromCQM
from transformations.from_bqm import FromBQM

class FromCPLEX:

    def __init__(self, cplex: CplexModel):
        self.__cplex_model = cplex
        self.tmp_lp_dir = Path(__file__).resolve().parent / "tmp"
        import warnings
        warnings.filterwarnings("ignore")

    @property
    def cplex_model(self):
        return self.__cplex_model

    def to_bqm(self, lagrange_multiplier: Union[None, float] = None) -> tuple:
        cqm = self.to_cqm()
        bqm, converter_bqm_cqm = FromCQM(cqm).to_bqm(lagrange_multiplier=lagrange_multiplier)
        return bqm, converter_bqm_cqm

    def to_cqm(self):
        lp_file_path = self.to_lp(self.tmp_lp_dir)
        cqm = FromLP(file_path=lp_file_path).to_cqm()
        return cqm
    
    def to_lp(self, file_path):
        self.cplex_model.export_as_lp(basename=str(file_path))
        return file_path
    
    def to_qp(self):
        qp = from_docplex_mp(self.__cplex_model)
        return qp 

    def to_qubo_qiskit(self):
        qp = self.to_qp()
        qubo_qiskit = FromQP(qp).to_qubo_qiskit()
        return qubo_qiskit

    def to_gurobi(self):
        qp = self.to_qp()
        gurobi_model = FromQP(qp).to_gurobi()
        return gurobi_model
    
    def to_gurobi_as_file(self):
        file_path = self.to_lp(self.tmp_lp_dir)
        gurobi_model = FromLP(file_path).to_gurobi()
        return gurobi_model
        
    def to_matrix(self, lagrange_multiplier: Union[None, float] = None) -> tuple:
        cqm = self.to_cqm()
        qubo_matrix, conv = FromCQM(cqm).to_matrix(lagrange_multiplier)
        return qubo_matrix, conv

    def to_matrix_with_dict(self, lagrange_multiplier: Union[None, float] = None) -> tuple:
        bqm, _ = self.to_bqm(lagrange_multiplier)
        qubo_matrix, variable_indices = FromBQM(bqm).to_matrix()
        return qubo_matrix, variable_indices















