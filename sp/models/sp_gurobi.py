import time

import gurobipy as gp
from abstract.models.abstract_model import AbstractModel
from sp.data.sp_data import SPData


class SPGurobi(AbstractModel):

    def __init__(self, data: SPData):
        self.data = data
        self.model = gp.Model("SP")
        self.model.Params.LogToConsole = 0
        self.model.Params.OutputFlag = 0
        self.variables = {}

        self.build_model()

    def build_model(self):
        # add variable
        for lidar in self.data.listLidar:
            var_name = f"x_{lidar[0]}_{lidar[1]}_{lidar[2]}_{lidar[3]}_{lidar[4]}"
            self.variables[lidar] = self.model.addVar(vtype=gp.GRB.BINARY, name=var_name)

        # minimize lidar number
        objective = sum(self.variables.values())
        self.model.setObjective(objective, gp.GRB.MINIMIZE)

        # cover every street point
        for sp in self.data.listStreetPoints:
            has_neighbour = False
            for v in self.data.G.neighbors(sp):
                has_neighbour = True
            if has_neighbour:
                term = sum(self.variables[v] for v in self.data.G.neighbors(sp))
                self.model.addLConstr(term >= 1)

    def solve(self, **config):

        TimeLimit = config.get("TimeLimit", 60)
        self.model.Params.TimeLimit = TimeLimit

        start_time = time.time()
        self.model.optimize()
        runtime = time.time() - start_time

        solution = {var.VarName: int(var.X) for var in self.model.getVars()}

        return {"solution": solution, "energy": 0, "runtime": runtime}