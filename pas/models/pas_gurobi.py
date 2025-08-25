import gurobipy as gp
from gurobipy import GRB, Var

from functools import cached_property
import time

from pas.data.pas_data import PASData


class GurobiPAS:

    def __init__(self, data: PASData):
        self.data = data
        self.model = gp.Model("PAS")
        self.model.Params.LogToConsole = 0
        self.model.Params.OutputFlag = 0
        self.variables = {}
        for j in range(self.data.j):
            for m in range(self.data.m):
                for n in range(self.data.j):
                    self.variables[(j, m, n)] = self.model.addVar(lb=0.0, ub=1.0, name=f'x_{j}_{m}_{n}')
                    self.variables[(j, m, n)].VType = GRB.BINARY
        self.constant_bits = dict()
        for j in range(self.data.j):
            eligible = self.data.eligible_machines[j]
            for m in range(self.data.m):
                if m not in eligible:
                    for n in range(self.data.j):
                        self.variables[(j, m, n)] = 0
                        self.constant_bits[(j, m, n)] = 0

        for m in range(self.data.m):
            max_jobs = 0
            for e in self.data.eligible_machines:
                if m in e:
                    max_jobs += 1
            for j in range(self.data.j):
                for seq in range(max_jobs, self.data.j):
                    self.variables[(j, m, seq)] = 0
                    self.constant_bits[(j, m, seq)] = 0
        self.build_model()

        self.solved = False

    def build_model(self):
        # set constraints
        self.c2_job_max_once()
        self.c3_max_one_job_at_a_time()
        self.c4_no_timesteps_skipped()
        self.c5_all_jobs_done_once()
        # set objective
        H = -self.job_values_expression() + self.data.alpha * self.setup_times_expression() + self.data.beta * self.normalize_times_expression()
        self.model.Params.TimeLimit = 1000
        self.model.setObjective(H, GRB.MINIMIZE)
        self.model.params.NonConvex = 2

    def solve(self, **config):
        TimeLimit = config.get("TimeLimit", 1)
        self.model.Params.TimeLimit = TimeLimit
        self.model.update()
        # print(f"Variables: {self.model.NumVars}")
        # print(f"Constraints: {self.model.NumConstrs}")
        # print(f"Non-Zero Elements: {self.model.NumNZs}")


        start_time = time.time()
        self.model.optimize()
        runtime = time.time() - start_time

        solution = {var.VarName: int(var.X) for var in self.model.getVars()}

        self.solved = True

        return {"solution": solution, "energy": 0, "runtime": runtime}

    @cached_property
    def solution(self):
        solution = {}
        for var, val in self.variables.items():
            if isinstance(val, Var):
                solution[var] = int(val.X)
            else:
                solution[var] = val
        return solution

    def c2_job_max_once(self):
        for j in range(self.data.j):
            self.model.addLConstr(
                sum([self.variables[(j, m, n)] for m in range(self.data.m) for n in range(self.data.j)]) <= 1)


    def c3_max_one_job_at_a_time(self):
        for n in range(self.data.j):
            for m in range(self.data.m):
                self.model.addLConstr(
                    sum([self.variables[(j, m, n)] for j in range(self.data.j)]) <= 1)


    def c4_no_timesteps_skipped(self):
        for n in range(self.data.j - 1):
            for m in range(self.data.m):
                self.model.addLConstr(
                    sum([self.variables[(j, m, n + 1)] for j in range(self.data.j)]) - sum(
                        [self.variables[(j, m, n)] for j in range(self.data.j)]) <= 0)

    def c5_all_jobs_done_once(self):
        for j in range(self.data.j):
            self.model.addLConstr(
                sum([self.variables[(j, m, n)] for n in range(self.data.j) for m in range(self.data.m)]) == 1)

    def normalize_times_expression(self):
        H2 = 0
        for m in range(self.data.m):
            H2 += gp.quicksum([self.data.processing_times[j] * self.variables[(j, m, n)] for j in range(self.data.j) for n in range(self.data.j)]) ** 2
        return H2
    
    def setup_times_expression(self):
        H0 = 0
        for m in range(self.data.m):
            for j in range(self.data.j):
                for n in range(self.data.j - 1):
                    for j1 in range(self.data.j):
                        if j != j1:
                            H0 += self.data.setup_times[j][j1] * self.variables[j, m, n] * self.variables[j1, m, n + 1]
        return H0

    def job_values_expression(self):
        H1 = gp.quicksum([self.data.job_values[j][m] * self.variables[j, m, n] for j in range(self.data.j) for m in range(self.data.m) for n in range(self.data.j)])
        return H1