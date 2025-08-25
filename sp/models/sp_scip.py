import time

import pyscipopt as scip

class SPScip:

    def __init__(self, data):
        self.data = data
        self.model = scip.Model()
        self.model.hideOutput(quiet = True)
        self.variables = {}

        self.build_model()

    def solve(self, **config):

        start_time = time.time()
        self.model.optimize()
        runtime = time.time() - start_time

        solution = {}
        for lidar, var in self.variables.items():
            var_name = f"x_{lidar[0]}_{lidar[1]}_{lidar[2]}_{lidar[3]}_{lidar[4]}"
            solution[var_name] = self.model.getVal(var)

        return {"solution": solution, "energy": 0, "runtime": runtime}

    def build_model(self):
        for lidar in self.data.listLidar:
            var_name = f"x_{lidar[0]}_{lidar[1]}_{lidar[2]}_{lidar[3]}_{lidar[4]}"
            self.variables[lidar] = self.model.addVar(vtype="B", name=var_name)
        
        self.model.setObjective(sum(self.variables.values()), "minimize")

        for sp in self.data.listStreetPoints:
            has_neighbour = False
            for v in self.data.G.neighbors(sp):
                has_neighbour = True
            if has_neighbour:
                term = sum(self.variables[v] for v in self.data.G.neighbors(sp))
                self.model.addCons(term >= 1)
