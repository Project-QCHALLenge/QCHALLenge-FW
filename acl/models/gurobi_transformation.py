# Can we delete this file?

from transformations.from_cplex import FromCPLEX
import re
import gurobipy as gp


def cplex_2_gurobi(cplex_model):
    gurobi_model = FromCPLEX(cplex_model.model).to_gurobi()
    #gurobi_model.display()
    return gurobi_model

def gurobi_optimization(gurobi_model):
    gurobi_model.optimize()

    if gurobi_model.status == gp.GRB.OPTIMAL:
        print("Optimal objective value:", gurobi_model.objVal)

        gur_dict = {}
        for var in gurobi_model.getVars():

            gur_dict.update({var.varName: int(var.x)})
        return gur_dict

    else:
        print("Optimization was stopped with status", gurobi_model.status)
