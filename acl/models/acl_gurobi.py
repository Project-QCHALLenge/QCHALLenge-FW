import time
import gurobipy as gp

from docplex.mp.model import Model as CplexModel

from acl.models.acl_cplex_linear import CplexLinACL
from acl.data.acl_data import ACLData

from transformations.from_cplex import FromCPLEX


class GurobiACL(CplexModel):
    """
    Class to transform a Cplex model to a Gurobi model
    """

    def __init__(self, data: ACLData):
        # the _d_var_exists and _old_sack_var variable have been omitted so that they can be defined by ACL_Linear
        cplex_model = CplexLinACL(data=data)
        self.__model = FromCPLEX(cplex_model.model).to_gurobi()
        self.__model.Params.LogToConsole = 0
        self.__model.Params.OutputFlag = 0

    def model(self):
        return self.__model

    def solve(self, **config):
        timelimit = config.get("TimeLimit", 1)
        self.__model.Params.TimeLimit = timelimit

        start_time = time.time()
        self.__model.optimize()

        if self.__model.status == gp.GRB.OPTIMAL:
            runtime = time.time() - start_time

            solution = {}
            for var in self.__model.getVars():
                solution.update({var.varName: int(var.x)})
            return {"solution": solution, "runtime": runtime}

        else:
            print("Optimization was stopped with status", self.__model.status)
