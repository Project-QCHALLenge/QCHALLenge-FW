import time
from abstract.models.abstract_model import AbstractModel
from docplex.mp.model import Model
from transformations import FromCPLEX


class CPlexSP(AbstractModel):
    def __init__(self, data) -> None:
        self.data=data
        self.model = Model(name="SP")
        self.build_model()


    def solve(self, **config):

        TimeLimit = config.get("TimeLimit", 60)
        self.model.set_time_limit(TimeLimit)

        start_time = time.time()
        self.model.solve()
        runtime = time.time() - start_time  

        solution = {var.name.replace("m", "-"): var.solution_value for var in self.model.iter_variables()}

        return {"solution": solution, "runtime": runtime}
    

    def solve_dict(self, **config):

        TimeLimit = config.get("TimeLimit", 1)
        self.model.set_time_limit(TimeLimit)
        self.model.solve()

        solution = {var.name.replace("m", "-"): var.solution_value for var in self.model.iter_variables()}

        return solution
    

    def solve_qubo(self, solve_func, **config):

        qubo, converter = FromCPLEX(self.model).to_matrix()

        start_time = time.time()
        answer = solve_func(Q=qubo, **config)
        runtime = time.time() - start_time  

        solution = converter(answer.first.sample)
        solution = {k.replace("m", "-"): v for k,v in solution.items()}
        
        return {"solution": solution, "energy": answer.first.energy, "runtime": runtime}


    def build_model(self):
        x = self.model.binary_var_dict(self.data.listLidar, name='x')
        self.model.objective_expr = sum(x[i] for i in self.data.listLidar)
        self.model.objective_sense = 'min'
        for node in self.data.listStreetPoints:
            has_neighbour=0
            for v in self.data.G.neighbors(node): 
                has_neighbour=1
            if has_neighbour:
                self.model.add_constraint(1 <= sum(x[v] for v in self.data.G.neighbors(node)))