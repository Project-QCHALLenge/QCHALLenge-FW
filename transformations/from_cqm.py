from typing import Union
from functools import partial
from pathlib import Path
from docplex.mp.model_reader import ModelReader
from dimod import ConstrainedQuadraticModel, cqm_to_bqm, lp
from qiskit_optimization.translators import from_docplex_mp

from transformations.from_bqm import FromBQM

class FromCQM:

    def __init__(self, cqm: ConstrainedQuadraticModel):
        self.tmp_lp_dir = Path(__file__).resolve().parent / "tmp"
        self.__cqm_model = cqm

    def to_bqm(self, lagrange_multiplier: Union[None, float] = None):
        bqm, converter_bqm_cqm = cqm_to_bqm(cqm=self.__cqm_model, lagrange_multiplier=lagrange_multiplier)
        return bqm, converter_bqm_cqm
    
    def to_cplex(self):
        file_path = self.to_lp(self.tmp_lp_dir)
        cplex = ModelReader.read(f'{file_path}.lp', ignore_names=True)
        return cplex
    
    def to_lp(self, file_path):
        with open(f'{file_path}.lp', 'wb') as f:
            f.write(lp.dumps(self.__cqm_model).encode('utf-8'))
        return file_path

    def to_qp(self):
        cplex_model = self.to_cplex()
        qp = from_docplex_mp(cplex_model)
        return qp
    
    def to_matrix(self, lagrange_multiplier: Union[None, float] = None) -> tuple:
        bqm, inverter_bqm = self.to_bqm(lagrange_multiplier)
        qubo_matrix, variable_indices = FromBQM(bqm).to_matrix()
        def inverter(sample, var_indices, inverter_bqm):
            # sample_list = list(sample.values())
            # sample_list = {key: sample_list[key] for key in sorted(sample_list)}
            var_sample = {name: sample[index] for name, index in var_indices.items()}
            return inverter_bqm(var_sample)

        return qubo_matrix, partial(inverter, var_indices=variable_indices, inverter_bqm=inverter_bqm)
