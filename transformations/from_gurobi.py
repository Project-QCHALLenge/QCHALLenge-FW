from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_docplex_mp, from_gurobipy
from gurobipy import Model as GurobiModel



class FromGUROBI:

    def __init__(self, gurobi_model: GurobiModel):
        self.gurobi_model = gurobi_model

    def to_cplex(self):
        qp = from_gurobipy(self.gurobi_model)
        docplex_model = to_docplex_mp(qp)
        return docplex_model

    def to_qubo_qiskit(self):
        qp = from_gurobipy(self.gurobi_model)
        qubo = QuadraticProgramToQubo().convert(qp)
        return qubo



