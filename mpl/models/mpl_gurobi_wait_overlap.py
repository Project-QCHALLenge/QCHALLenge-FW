from gurobipy import GRB
import gurobipy as gp
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from abstract.models.abstract_model import AbstractModel
warnings.simplefilter(action="ignore", category=FutureWarning)


class MPLGurobiWaitOverlap(AbstractModel):
    def __init__(self, data):
        # processing / production time for different job types on different machines
        data = data.__dict__
        self.processing_times = data["processing_times"]

        # Machines
        self.M = data["M"]
        self.machine_names = data["machine_names"]

        # time for AGV move
        self.t_r = data["t_r"]
        # number of A and B type jobs
        self.N_A = data["N_A"]
        self.N_B = data["N_B"]
        # total time
        self.T = data["T"]
        # number of AGVs
        self.R = data["R"]
        # timelimit for solving
        self.timelimit = data["timelimit"]
        # set up with the parameters above
        self.JOBS_A = list(range(self.N_A))
        self.JOBS_B = list(range(self.N_A, self.N_A + self.N_B))

        self.p_a = [
            self.processing_times["Jobs_A"][0],
            self.processing_times["Jobs_A"][1],
            0,
            0,
        ]
        self.p_b = [
            0,
            self.processing_times["Jobs_B"][0],
            self.processing_times["Jobs_B"][1],
            self.processing_times["Jobs_B"][2],
        ]

        self.TASKS = [0, 1]  # keeping (0), lifting (1)
        self.AGVS = list(range(self.R))
        self.MACHINES = list(range(self.M))
        self.total_machines = len(self.MACHINES)
        self.total_jobs = len(self.JOBS_A) + len(self.JOBS_B)

        self.p = [self.p_a] * len(self.JOBS_A)
        self.p.extend([self.p_b] * len(self.JOBS_B))

        # set up model
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        self.model = gp.Model("mip", env=env)
        self.model.Params.LogToConsole = 0
        self.model.Params.OutputFlag = 0

        # define variables
        self.x = dict()
        for j in self.JOBS_A + self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for m in self.MACHINES:
                        for t in range(self.T):
                            self.x[(j, d, r, m, t)] = self.model.addVar(
                                lb=0.0, ub=1.0, name=f"x_{j}_{d}_{r}_{m}_{t}"
                            )
                            self.x[(j, d, r, m, t)].VType = GRB.BINARY

        # add variables for endpoint such that makespan can be minimized for
        self.u = dict()
        for t in range(self.T):
            self.u[t] = self.model.addVar(lb=0.0, ub=1.0, name=f"u_{t}")
            self.u[t].VType = GRB.BINARY

        # reduce number of variables by constants
        self.num_rem_vars, self.constant_bits = self.add_constants()

        self.solved = False
        self.build_model()

    def build_model(self):
        # add constraints
        self.constraint1()
        self.constraint2()
        self.constraint3()
        self.constraint4()
        self.constraint5()
        self.constraint6()
        self.constraint7()
        self.constraint8()
        self.constraint9()
        self.constraint10()
        self.constraint_u()
        # set objective
        self.add_objective()
        # set model Params
        self.model.Params.TimeLimit = self.timelimit
        self.model.Params.LogToConsole = 0
        self.model.Params.OutputFlag = 0

    def build_model_selected_constraints(self, constraints_list, solution_schedule):
        self.model.Params.LogToConsole = 0
        self.model.Params.OutputFlag = 0
        # add constraints that are in constraints_list
        if 1 in constraints_list:
            self.constraint1()
        if 2 in constraints_list:
            self.constraint2()
        if 3 in constraints_list:
            self.constraint3()
        if 4 in constraints_list:
            self.constraint4()
        if 5 in constraints_list:
            self.constraint5()
        if 6 in constraints_list:
            self.constraint6()
        if 7 in constraints_list:
            self.constraint7()
        if 8 in constraints_list:
            self.constraint8()
        if 9 in constraints_list:
            self.constraint9()
        if 10 in constraints_list:
            self.constraint10()

        self.add_solution_constraint(solution_schedule)

        # set objective
        self.add_objective()
        # set model Params, use timelimit of 1000 only for constraints check
        self.model.Params.TimeLimit = 1000
        self.model.Params.LogToConsole = 0
        self.model.Params.OutputFlag = 0

    def solve_model(self):
        self.model.optimize()
        self.solved = True

    def solve(self, **params):

        self.model.update()
        # print("Number of variables:", self.model.NumVars)
        # print("Number of constraints:", self.model.NumConstrs)
        
        timelimit = params.get("TimeLimit", False)
        if timelimit:
            self.model.Params.TimeLimit = self.timelimit

        self.model.optimize()
        t_endpoint, schedule, pos_vars = self.get_solution()
        answer = {}

        # print(self.model.status)
        if self.model.status == 3:
            energy = -1
            solution = {var.VarName: 0 for var in self.model.getVars()}
        else:
            energy =  self.model.objVal
            solution = {var.VarName: int(var.X) for var in self.model.getVars()}        

        answer["objVal"] = energy
        answer["status"] =  self.model.status
        answer["runtime"] = self.model.RunTime
        answer["t_endpoint"] =  t_endpoint
        answer["schedule"] = schedule
        answer["pos_vars"] = pos_vars
        answer["solution"] = solution
        answer["p"] = self.p
        answer["machine_names"] = self.machine_names
        answer["total_machines"] = self.total_machines
        return answer

    def get_solution(self):
        schedule = dict()
        pos_vars = []
        if self.model.status != 3:
            for j in self.JOBS_A + self.JOBS_B:
                for m in self.MACHINES:
                    for d in self.TASKS:
                        for t in range(self.T):
                            for r in self.AGVS:
                                if type(self.x[(j, d, r, m, t)]) != int:
                                    if self.x[(j, d, r, m, t)].X == 1.0:
                                        schedule[(j, m, d)] = (t, r)
                                        pos_vars.append(self.x[(j, d, r, m, t)])
            # checking for max time
            t_endpoint = 0
            for t in range(self.T):
                if self.u[t].X == 1.0:
                    t_endpoint = t
        else:
            t_endpoint = "inf"

        return t_endpoint, schedule, pos_vars

    def save_results(self, filename):
        columns = [
            "Gurobi variable",
            "Variable name",
            "Variable value",
        ]

        results = pd.DataFrame(columns=columns)

        for var in self.model.getVars():
            row = [var, var.VarName, var.X]
            results.loc[len(results)] = row

        results_path = (Path.cwd().parent / "results" / filename).resolve()
        results.to_csv(results_path, sep=",")

    def add_objective(self):
        # H = sum([t * self.u[t] for t in range(self.T)]) + sum([self.x[(j, 0, r, m, t)] for j in self.JOBS_A
        # + self.JOBS_B for r in self.AGVS for m in self.MACHINES for t in range(self.T)])
        # modified to avoid objective becoming too big
        H = sum([t * self.u[t] for t in range(self.T)])
        self.model.setObjective(H, GRB.MINIMIZE)

    # obj value needs to be updated MANUALLY! below
    def get_objective_value(self, schedule, endpoints):
        endpoints_indicator = np.zeros(self.T)
        endpoints_indicator[endpoints] = 1
        u_vec = endpoints_indicator * [t for t in range(self.T)]

        return sum(u_vec)

    def get_model(self):
        self.model.update()
        return self.model

    # Constraints
    def constraint1(self):
    # All A Jobs must be lifted from machine M1 once-and-only-once, 
    # by some AGV, at some point of time
        for j in self.JOBS_A:
            self.model.addLConstr(
                sum([self.x[(j, 1, r, 0, t)] for t in range(self.T) for r in self.AGVS])
                == 1
            )

    def constraint2(self):
    # All A and B Jobs must be kept on and lifted from machine M2 once-and-only-once, 
    # by some AGV, at some point of time
        for j in self.JOBS_A + self.JOBS_B:
            for d in self.TASKS:
                self.model.addLConstr(
                    sum(
                        [
                            self.x[(j, d, r, 1, t)]
                            for t in range(self.t_r, self.T)
                            for r in self.AGVS
                        ]
                    )
                    == 1
                )

    def constraint3(self):
    # All B Jobs must be kept-on and lifted-from machines M 3 once-and-only-once, 
    # by some AGV, at some point of time
        for j in self.JOBS_B:
            for d in self.TASKS:
                self.model.addLConstr(
                    sum(
                        [
                            self.x[(j, d, r, 2, t)]
                            for t in range(
                                self.t_r
                                + self.processing_times["Jobs_B"][0]
                                + self.t_r,
                                self.T,
                            )
                            for r in self.AGVS
                        ]
                    )
                    == 1
                )

    def constraint4(self):
    # All B Jobs must be kept-on and lifted-from machine M 4 once-and-only-once, 
    # by some AGV, at some points of time
        for j in self.JOBS_B:
            for d in self.TASKS:
                self.model.addLConstr(
                    sum(
                        [
                            self.x[(j, d, r, 3, t)]
                            for t in range(
                                self.t_r
                                + self.processing_times["Jobs_B"][0]
                                + self.t_r
                                + self.processing_times["Jobs_B"][1]
                                + self.t_r,
                                self.T,
                            )
                            for r in self.AGVS
                        ]
                    )
                    == 1
                )

    def constraint5(self):
    # An AGV can only perform at most one task at any given time
        for r in self.AGVS:
            for t in range(self.T):
                self.model.addLConstr(
                    sum(
                        [
                            self.x[(j, d, r, m, t)]
                            for j in self.JOBS_A + self.JOBS_B
                            for d in self.TASKS
                            for m in self.MACHINES
                        ]
                    )
                    <= 1
                )

    def constraint6(self):
        # f a job j has been lifted from a machine m at some time t by an AGV, 
        # then the same job j can not have been kept on the machine m at any time 
        # between t − pm j + 1 to T − 1 by any AGV
        # For Job A we need this constraint only for M2 (m=1)
        for t in range(self.T):
            for j in self.JOBS_A:
                for m in [1]:
                    start_time = max(0, t - self.p_a[m] + 1)
                    for r in self.AGVS:
                        for t1 in range(start_time, self.T):
                            for r1 in self.AGVS:
                                self.model.addLConstr(
                                    sum(
                                        [
                                            self.x[(j, 1, r, m, t)]
                                            + self.x[(j, 0, r1, m, t1)]
                                        ]
                                    )
                                    <= 1
                                )

        for t in range(self.T):
            for m in [1, 2, 3]:
                for j in self.JOBS_B:
                    start_time = max(0, t - self.p_b[m] + 1)
                    for r in self.AGVS:
                        for t1 in range(start_time, self.T):
                            for r1 in self.AGVS:
                                self.model.addLConstr(
                                    sum(
                                        [
                                            self.x[(j, 1, r, m, t)]
                                            + self.x[(j, 0, r1, m, t1)]
                                        ]
                                    )
                                    <= 1
                                )

    def constraint7(self):
        # If a job j has been kept on a machine m at some time t by an AGV, 
        # then it can not be lifted from that machine at any time 
        # between 0 to t + pm j − 1 by any AGV
        # For Job A we need this constraint only for M2 (m=1)
        for t in range(self.T):
            for j in self.JOBS_A:
                for m in [1]:
                    end_time = min(t + self.p_a[m] - 1, self.T - 1)
                    for r in self.AGVS:
                        for t1 in range(end_time + 1):
                            for r1 in self.AGVS:
                                self.model.addLConstr(
                                    sum(
                                        [
                                            self.x[(j, 0, r, m, t)]
                                            + self.x[(j, 1, r1, m, t1)]
                                        ]
                                    )
                                    <= 1
                                )

        for t in range(self.T):
            for m in [1, 2, 3]:
                for j in self.JOBS_B:
                    end_time = min(t + self.p_b[m] - 1, self.T - 1)
                    for r in self.AGVS:
                        for t1 in range(end_time + 1):
                            for r1 in self.AGVS:
                                self.model.addLConstr(
                                    sum(
                                        [
                                            self.x[(j, 0, r, m, t)]
                                            + self.x[(j, 1, r1, m, t1)]
                                        ]
                                    )
                                    <= 1
                                )

    def constraint8(self):
        # If a B type job j has been kept on a machine m + 1 at some time t by an AGV, 
        # then it can not have been lifted from the machine m at any time 
        # between t − rt + 1 to T by any AGV 
        # (the second placement of B jobs onto Machine M2 is modelled 
        # as placement onto a hypothetical Machine M4)
        for t in range(self.T):
            for m in [1, 2]:
                for j in self.JOBS_B:
                    start_time = max(t - self.t_r + 1, 0)
                    for r in self.AGVS:
                        for t1 in range(start_time, self.T):
                            for r1 in self.AGVS:
                                self.model.addLConstr(
                                    sum(
                                        [
                                            self.x[(j, 0, r, m + 1, t)]
                                            + self.x[(j, 1, r1, m, t1)]
                                        ]
                                    )
                                    <= 1
                                )

        for t in range(self.T):
            for m in [0]:
                for j in self.JOBS_A:
                    start_time = max(t - self.t_r + 1, 0)
                    for r in self.AGVS:
                        for t1 in range(start_time, self.T):
                            for r1 in self.AGVS:
                                self.model.addLConstr(
                                    sum(
                                        [
                                            self.x[(j, 0, r, m + 1, t)]
                                            + self.x[(j, 1, r1, m, t1)]
                                        ]
                                    )
                                    <= 1
                                )

    def constraint9(self):
        # At any point of time t, the number of jobs kept on a machine m 
        # between 0 to t must be either equal to the number of jobs lifted 
        # from the machine m between 0 to t, or the number of jobs kept on
        # the machine between 0 to t must be one more than the number of jobs 
        # lifted from the machine between 0 to t. 
        # This constraint helps us in ensuring that any job j′ 
        # can not be kept or lifted on machine m in between the
        # time-window of some other jobs’ keeping and lifting, on machine m
        for m in [2]:
            for t in range(self.T):
                first_term = sum(
                    [
                        self.x[(j, 0, r, m, t1)]
                        for j in self.JOBS_A + self.JOBS_B
                        for t1 in range(t + 1)
                        for r in self.AGVS
                    ]
                )
                second_term = sum(
                    [
                        self.x[(j, 1, r1, m, t1)]
                        for j in self.JOBS_A + self.JOBS_B
                        for t1 in range(t + 1)
                        for r1 in self.AGVS
                    ]
                )
                self.model.addLConstr(first_term - second_term <= 1)
                self.model.addLConstr(first_term - second_term >= 0)

        for t in range(self.T):
            first_term = sum(
                [
                    self.x[(j, 0, r, m, t1)]
                    for j in self.JOBS_A + self.JOBS_B
                    for t1 in range(t + 1)
                    for r in self.AGVS
                    for m in [1, 3]
                ]
            )
            second_term = sum(
                [
                    self.x[(j, 1, r1, m, t1)]
                    for j in self.JOBS_A + self.JOBS_B
                    for t1 in range(t + 1)
                    for r1 in self.AGVS
                    for m in [1, 3]
                ]
            )
            self.model.addLConstr(first_term - second_term <= 1)
            self.model.addLConstr(first_term - second_term >= 0)

    def constraint10(self):
        # If a job i has been kept on a machine x at time t by an AGV r, 
        # then any job can not be kept on any machine 
        # at time t + 2 · rt − 1, by the same AGV r
        for r in self.AGVS:
            for t in range(self.T):
                for m in [0, 1, 2, 3]:
                    for m1 in [0, 1, 2, 3]:
                        for i in self.JOBS_A + self.JOBS_B:
                            for j in self.JOBS_A + self.JOBS_B:
                                if i != j:
                                    for bot_time in range(1, 2 * self.t_r):
                                        if t + bot_time <= self.T - 1:
                                            self.model.addLConstr(
                                                sum(
                                                    [
                                                        self.x[(i, 0, r, m, t)]
                                                        + self.x[
                                                            (j, 0, r, m1, t + bot_time)
                                                        ]
                                                    ]
                                                )
                                                <= 1
                                            )
        for r in self.AGVS:
            for t in range(self.T):
                for m in [0, 1, 2, 3]:
                    for m1 in [0, 1, 2, 3]:
                        for i in self.JOBS_A + self.JOBS_B:
                            for j in self.JOBS_A + self.JOBS_B:
                                if i != j:
                                    for bot_time in range(2 * self.t_r):
                                        if t + bot_time <= self.T - 1:
                                            self.model.addLConstr(
                                                sum(
                                                    [
                                                        self.x[(i, 1, r, m, t)]
                                                        + self.x[
                                                            (j, 1, r, m1, t + bot_time)
                                                        ]
                                                    ]
                                                )
                                                <= 1
                                            )

        for r in self.AGVS:
            for t in range(self.T):
                for m in [0, 1, 2, 3]:
                    for m1 in [0, 1, 2, 3]:
                        for j in self.JOBS_A + self.JOBS_B:
                            for bot_time in range(self.t_r):
                                if t + bot_time <= self.T - 1:
                                    self.model.addLConstr(
                                        sum(
                                            [
                                                self.x[(j, 1, r, m, t)]
                                                + self.x[(j, 0, r, m1, t + bot_time)]
                                            ]
                                        )
                                        <= 1
                                    )

        for r in self.AGVS:
            for t in range(self.T):
                for m in [0, 1, 2, 3]:
                    for m1 in [0, 1, 2, 3]:
                        for j in self.JOBS_A + self.JOBS_B:
                            for j1 in self.JOBS_A + self.JOBS_B:
                                if j != j1:
                                    for bot_time in range(2 * self.t_r):
                                        if t + bot_time <= self.T - 1:
                                            self.model.addLConstr(
                                                sum(
                                                    [
                                                        self.x[(j, 1, r, m, t)]
                                                        + self.x[
                                                            (j1, 0, r, m1, t + bot_time)
                                                        ]
                                                    ]
                                                )
                                                <= 1
                                            )

        for r in self.AGVS:
            for t in range(self.T):
                for m in [0, 1, 2, 3]:
                    for m1 in [0, 1, 2, 3]:
                        # new condition to prevent min time gap if AGV acts on same machine
                        # if m != m1: taken out as only relevant for t_r similar to processing times and the sequence of
                        # keep, then lift on the same machine
                        for j in self.JOBS_A + self.JOBS_B:
                            for j1 in self.JOBS_A + self.JOBS_B:
                                for bot_time in range(2 * self.t_r):
                                    if t + bot_time <= self.T - 1:
                                        self.model.addLConstr(
                                            sum(
                                                [
                                                    self.x[(j, 0, r, m, t)]
                                                    + self.x[
                                                        (j1, 1, r, m1, t + bot_time)
                                                    ]
                                                ]
                                            )
                                            <= 1
                                        )

        for r1 in self.AGVS:
            for r2 in self.AGVS:
                if r1 != r2:
                    for t in range(self.T):
                        for m in [0, 1, 2, 3]:
                            for m1 in [0, 1, 2, 3]:
                                for j in self.JOBS_A + self.JOBS_B:
                                    for bot_time in range(2 * self.t_r):
                                        if t + bot_time <= self.T - 1:
                                            self.model.addLConstr(
                                                sum(
                                                    [
                                                        self.x[(j, 1, r1, m, t)]
                                                        + self.x[
                                                            (j, 0, r2, m1, t + bot_time)
                                                        ]
                                                    ]
                                                )
                                                <= 1
                                            )

    # Constraint for makespan endpoint
    def constraint_u(self):
        self.model.addLConstr(sum([self.u[t] for t in range(self.T)]) == 1)
        for j in self.JOBS_A + self.JOBS_B:
            for r in self.AGVS:
                for m in [0, 1, 2, 3]:
                    for t1 in range(1, self.T):
                        for t2 in range(t1 + 1):
                            self.model.addLConstr(
                                sum([self.x[(j, 1, r, m, t1)] + self.u[t2]]) <= 1
                            )

    def add_constants(self):
        constants = dict()

        # No B jobs need to be kept/lifted on machine M1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(self.T):
                        self.x[(j, d, r, 0, t)] = 0
                        constants[(j, d, r, 0, t)] = 0

        # A jobs need not be kept on machine M1 ( they are assumed to be kept automatically)
        for j in self.JOBS_A:
            for r in self.AGVS:
                for t in range(self.T):
                    self.x[(j, 0, r, 0, t)] = 0
                    constants[(j, 0, r, 0, t)] = 0

        # The A jobs on machine M1 cannot be lifted before they are processed on M1
        for j in self.JOBS_A:
            consumed_time = (j + 1) * self.processing_times["Jobs_A"][0]
            for r in self.AGVS:
                for t in range(0, consumed_time):
                    self.x[(j, 1, r, 0, t)] = 0
                    constants[(j, 1, r, 0, t)] = 0

        # The A jobs on machine M1 cannot be lifted after the processing of the next job on M1
        for j in self.JOBS_A:
            final_lifting_time = (j + 2) * self.processing_times["Jobs_A"][0]
            for r in self.AGVS:
                for t in range(final_lifting_time, self.T):
                    self.x[(j, 1, r, 0, t)] = 0
                    constants[(j, 1, r, 0, t)] = 0

        # The A jobs on machine M2 cannot be kept before they are processed on M1 and transported to M2
        for j in self.JOBS_A:
            for d in self.TASKS:
                consumed_time = (
                    (j + 1) * self.processing_times["Jobs_A"][0]
                    + self.t_r
                    + d * (self.processing_times["Jobs_A"][1])
                )
                for r in self.AGVS:
                    for t in range(0, consumed_time):
                        self.x[(j, d, r, 1, t)] = 0
                        constants[(j, d, r, 1, t)] = 0

        # no A jobs need to be processed on M3 and M4
        for j in self.JOBS_A:
            for d in self.TASKS:
                for r in self.AGVS:
                    for m in [2, 3]:
                        for t in range(self.T):
                            self.x[(j, d, r, m, t)] = 0
                            constants[(j, d, r, m, t)] = 0

        # no B jobs can be kept on machine M2 from time t=0 to t=t_r-1, and lifted from t=0 to t=t_r+p_j^m-1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(self.t_r + d * (self.processing_times["Jobs_B"][0])):
                        self.x[(j, d, r, 1, t)] = 0
                        constants[(j, d, r, 1, t)] = 0

        # no job B can be kept on machine M3 from t=0 and t=t_r+p_j^m+r_t-1 and lifted from M3
        # between time t=0 to t=t_r+p_b^0+t_r-1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(
                        self.t_r
                        + self.processing_times["Jobs_B"][0]
                        + self.t_r
                        + d * (self.processing_times["Jobs_B"][1])
                    ):
                        self.x[(j, d, r, 2, t)] = 0
                        constants[(j, d, r, 2, t)] = 0

        # no job B can be kept on machine M4 from t=0 and t=t_r+p_j^m+r_t-1 and lifted from M3
        # between time t=0 to t=t_r+p_b^0+t_r-1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(
                        self.t_r
                        + self.processing_times["Jobs_B"][0]
                        + self.t_r
                        + self.processing_times["Jobs_B"][1]
                        + self.t_r
                        + d * (self.processing_times["Jobs_B"][2])
                    ):
                        self.x[(j, d, r, 3, t)] = 0
                        constants[(j, d, r, 3, t)] = 0

        # remaining variables
        rem_vars = (
            (self.N_A + self.N_B) * 2 * self.R * self.M * self.T
            - len(constants)
            + self.T
        )
        return rem_vars, constants

    # add constraint that value for the solution are 1 and 0 for all other variables
    def add_solution_constraint(self, solution_schedule):
        solution_list = self.create_solution_list(solution_schedule)

        for j in self.JOBS_A + self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for m in self.MACHINES:
                        for t in range(self.T):
                            if (j, d, r, m, t) in solution_list:
                                self.model.addLConstr(self.x[(j, d, r, m, t)] == 1)
                            else:
                                self.model.addLConstr(self.x[(j, d, r, m, t)] == 0)

    # pull solution indices out of schedule in the right order
    def create_solution_list(self, solution_schedule):
        sol_list = []

        for key, value in solution_schedule.items():
            j = key[0]
            m = key[1]
            d = key[2]
            t = value[0]
            r = value[1]
            sol_list.append((j, d, r, m, t))

        return sol_list
