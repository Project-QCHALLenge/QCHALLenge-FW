import warnings
from lb.models.lb_abstract_model import LBAbstractModel
from lb.data.lb_data import LBData
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import itertools
from lb.models.lb_qubo_uniform_truck_capacity import LBQUBOUnaryUniformTruckCapacity
from abstract.models.abstract_model import AbstractModel
warnings.simplefilter(action='ignore', category=FutureWarning)


class LBGurobiQUBOUnaryUniformTruckCapacity(LBAbstractModel, AbstractModel):

    def _create_variables(self):
        self.decision_variables = self.model.addVars(self.valid_indices,
                                                     vtype=GRB.BINARY, name="x")

        self.decision_variables_vector = gp.MVar.fromlist(list(self.decision_variables.values()))

    def _create_data_structures(self):
        self.valid_indices = itertools.product(self.trucks, self.units)
    def solve(self, **params):
        self.model.optimize()

    def solution(self):
        solution = {"solution" : {}}
        replacement_characters = str.maketrans({"[":"_", "]":"", ",":"_"})
        if self.model.Status != GRB.INFEASIBLE:
            for index, var in self.decision_variables.items():
                var_name_with_underscores = str(var.varName).translate(replacement_characters)
                solution["solution"][var_name_with_underscores] = var.X
        solution["energy"] = float(self.model.getObjective().getValue())
        solution["runtime"] = self.model.Runtime
        return solution

    def build_model(self):
        self.Q, self.c, _ = LBQUBOUnaryUniformTruckCapacity(self.data).assemble()
        x = self.decision_variables_vector
        self.model.setObjective(x.T @ self.Q @ x + self.c, GRB.MINIMIZE)


    def set_initial_solution(self, solution):
        for x in solution:
            self.decision_variables[x].Start = 1
            #self.model.addConstr(self.decision_variables[x] == 1)

    def __init__(self, data: LBData):
        super().__init__(data)
        self.model = gp.Model("LB")
        self.Q = None
        self.decision_variables = None
        self.loads = None
        self.units = range(self.data.number_of_units)
        self.product_type_to_indices = self.data.product_type_to_indices
        self.trucks = range(self.data.number_of_trucks)
        self.weights = self.data.weights
        self.valid_indices = None
        self.product_types = self.data.product_types

        self._create_data_structures()
        self._create_variables()
        self.build_model()


