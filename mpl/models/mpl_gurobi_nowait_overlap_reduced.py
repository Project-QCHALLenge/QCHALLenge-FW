# -*- coding: utf-8 -*-

import time
import gurobipy as gp
import numpy as np
import pandas as pd
from itertools import product
from abstract.models.abstract_model import AbstractModel


class MPLGurobiNoWaitOverlapReduced(AbstractModel):
    def __init__(self, data, overlap = True):

        self.OVERLAP = overlap

        self.processing_times_A = data.processing_times["Jobs_A"]
        self.processing_times_B = data.processing_times["Jobs_B"]
        self.n_A_jobs = data.N_A
        self.n_B_jobs = data.N_B
        self.n_AGVs = data.R
        self.delta = data.t_r
        
        self.T = data.T
        self.J = self.n_A_jobs + 3*self.n_B_jobs
        p = np.zeros(self.J)
        # processing time of A jobs
        p[:self.n_A_jobs] = self.processing_times_A[1]
        # processing time of B jobs
        self.n_A_jobs
        p[self.n_A_jobs:self.n_A_jobs+self.n_B_jobs] = self.processing_times_B[0]
        p[self.n_A_jobs+self.n_B_jobs:self.n_A_jobs+2*self.n_B_jobs] = self.processing_times_B[1]
        p[self.n_A_jobs+2*self.n_B_jobs:self.n_A_jobs+3*self.n_B_jobs] = self.processing_times_B[2]
        # make processing time integral for indexing
        self.p = p.astype(int)

        self.start = list(range(self.n_A_jobs,self.n_A_jobs+self.n_B_jobs)) + list(range(self.n_A_jobs+self.n_B_jobs,self.n_A_jobs+2*self.n_B_jobs))
        self.end = list(range(self.n_A_jobs+self.n_B_jobs,self.n_A_jobs+2*self.n_B_jobs)) + list(range(self.n_A_jobs+2*self.n_B_jobs,self.n_A_jobs+3*self.n_B_jobs))

        self.machines = [
            [j for j in range(self.n_A_jobs)] + [j for j in range(self.n_A_jobs,self.n_A_jobs+self.n_B_jobs)] + [j for j in range(self.n_A_jobs+2*self.n_B_jobs,self.n_A_jobs+3*self.n_B_jobs)], # competition for machine 2
            [j for j in range(self.n_A_jobs+self.n_B_jobs,self.n_A_jobs+2*self.n_B_jobs)] # competition for machine 3
        ]

        self.model = gp.Model()
        self.model.Params.TimeLimit = 1000
        self.model.Params.LogToConsole = 0
        self.model.Params.OutputFlag = 0

        self.x = self.model.addVars(self.T,self.J,self.n_AGVs,self.n_AGVs,vtype=gp.GRB.BINARY,name='x')
        self.y = self.model.addVars(self.n_A_jobs,self.n_AGVs,vtype=gp.GRB.BINARY,name='y') # model AGV pick-up A jobs
        self.c = self.model.addVar(obj=1,name='c')
        self.x_const = {}

        self.build_model()

    def clip(self, t):
        return int(np.clip(t,0,self.T))

    def solve(self, **params):

        #self.model.update()
        #print("Number of variables:", self.model.NumVars)
        #print("Number of constraints:", self.model.NumConstrs)
        
        timelimit = params.get("TimeLimit", False)
        if timelimit:
            self.model.Params.TimeLimit = timelimit

        self.model.optimize()
        #print(self.model.status)
        if self.model.status == 3:
            energy = -1
            solution = {var.VarName: 0 for var in self.model.getVars()}
            schedule = {}
            x_vals = {}
        else:
            energy =  self.model.objVal
            solution = {var.VarName: int(var.X) for var in self.model.getVars()}  

        for key, value in self.x_const.items():
            formatted_key = f"x_{key[0]}_{key[1]}_{key[2]}_{key[3]}"
            solution[formatted_key] = value

        converted_solution = {}
        for key, value in solution.items():
            new_key = key.replace('[', '_').replace(']', '').replace(',', '_')
            converted_solution[new_key] = value

        answer = {}
        answer["objVal"] = energy
        answer["runtime"] = self.model.RunTime
        answer["solution"] = converted_solution
        return answer
    
    def build_model(self):
        
        # All jobs must be transported to and from the machine by one AGV:
        self.model.addConstrs(
            gp.quicksum(self.x[t,j,k1,k2] for t in range(self.T) for k1 in range(self.n_AGVs) for k2 in range(self.n_AGVs)) == 1 for j in range(self.J)
        )

        # All first A jobs must be picked up:
        self.model.addConstrs(
            gp.quicksum(self.y[j,k] for k in range(self.n_AGVs)) == 1 for j in range(self.n_A_jobs)
        )

        # The last finished job defines the make span:
        last_jobs = [j for j in range(self.n_A_jobs)] + [j for j in range(self.n_A_jobs+2*self.n_B_jobs,self.J)]
        self.model.addConstrs(
            gp.quicksum((t+self.p[j])*self.x[t,j,k1,k2] for t in range(self.T) for k1 in range(self.n_AGVs) for k2 in range(self.n_AGVs)) <= self.c for j in last_jobs
        )

        # At each time t only one job can be processed per machine:
        self.model.addConstrs(
            gp.quicksum(self.x[s,j,k1,k2] for j in self.machines[m] for s in range(max(t-self.p[j]+1,0),t+1) for k1 in range(self.n_AGVs) for k2 in range(self.n_AGVs)) <= 1 for t in range(self.T) for m in range(len(self.machines))
        )

        # A job can only be started when the predecessor finished
        transportation = self.delta if self.OVERLAP else 2*self.delta
        self.model.addConstrs(gp.quicksum((t+self.p[i]+transportation)*self.x[t,i,k1,k2] for t in range(self.T) for k1 in range(self.n_AGVs) for k2 in range(self.n_AGVs)) <= gp.quicksum(t*self.x[t,j,k1,k2] for t in range(self.T) for k1 in range(self.n_AGVs) for k2 in range(self.n_AGVs)) for i,j in zip(self.start,self.end))
        
        # For each time step t no two jobs can start during t-delta+1, ... ,t+\delta\ using the same AGV k
        self.model.addConstrs(gp.quicksum(self.x[s,j,k,k2] for k2 in range(self.n_AGVs) for j in range(self.J) for s in range(self.clip(t-self.delta+1),self.clip(t+self.delta+1))) <= 1 for t in range(self.T) for k in range(self.n_AGVs))

        # For each time step t no two jobs can end during t-delta+1, ... ,t+\delta\ using the same AGV k
        def ending_jobs(t,k):
            return gp.quicksum(self.x[s,j,k1,k] for k1 in range(self.n_AGVs) for j in range(self.J) for s in range(self.clip(t-self.p[j]-self.delta+1),self.clip(t-self.p[j]+self.delta+1)))
        def ending_A_jobs(t,k):
            ending_jobs = [j for j in range(self.n_A_jobs) if t >= (j+1)*self.processing_times_A[0]-self.delta and t <= (j+1)*self.processing_times_A[0]+self.delta-1]
            return gp.quicksum(self.y[j,k] for j in ending_jobs)
        t_min = self.processing_times_B[0]
        self.model.addConstrs(ending_jobs(t,k) + ending_A_jobs(t,k) <= 1 for t in range(t_min,self.T) for k in range(self.n_AGVs))

        # If job j starts at t and uses AGV k, another job i cannot finish in t-1 and be picked up by AGV k
        def starting_jobs(t,k):
            return gp.quicksum(self.x[t,j,k,k2] for j in range(self.J) for k2 in range(self.n_AGVs))
        def conflicting_ending_jobs(t,k):
            return gp.quicksum(self.x[self.clip(t-self.p[j]),j,k1,k] for j in range(self.J) for k1 in range(self.n_AGVs))
        self.model.addConstrs(starting_jobs(t,k) + conflicting_ending_jobs(t,k) <= 1 for t in range(self.T) for k in range(self.n_AGVs))

        # If a job j ends at t no job can start in t-delta+1, ... , t and use the same AGV for transporting to the machine.
        def ending_jobs(t,k):
            return gp.quicksum(self.x[self.clip(t-self.p[j]),j,k1,k] for j in range(self.J) for k1 in range(self.n_AGVs))
        def ending_A_jobs(t,k):
            if t == 0 or t > self.n_A_jobs*self.processing_times_A[0]:
                return 0
            if t % self.processing_times_A[0] == 0:
                j = t//self.processing_times_A[0]-1
                return self.y[j,k]
            return 0
        def conflicting_jobs(t,k):
            return gp.quicksum(self.x[self.clip(s),j,k,k2] for j in range(self.J) for s in range(t-self.delta+1,t+1) for k2 in range(self.n_AGVs))
        self.model.addConstrs(ending_jobs(t,k)+ending_A_jobs(t,k)+conflicting_jobs(t,k) <= 1 for t in range(t_min,self.T) for k in range(self.n_AGVs))

        # If a job j starts at t no job but a direct predecessor of j can start within t-p_j-delta, ... , t-p_j+delta-1 and use the same AGV for transporting to the machine.
        non_conflicting = {j:[j] for j in range(self.J)}
        for i,j in zip(self.start,self.end):
            non_conflicting[j].append(i)
        def starting_jobs(t,j,k):
            return gp.quicksum(self.x[t,j,k,k2] for k2 in range(self.n_AGVs))
        def conflicting_A_jobs(t,j,k):
            conflicting_jobs = [i for i in range(self.n_A_jobs) if i != j and t > (i+1)*self.processing_times_A[0] and t <= (i+1)*self.processing_times_A[0]+2*self.delta]
            return gp.quicksum(self.y[i,k] for i in conflicting_jobs)
        def conflicting_jobs(t,j,k):
            return gp.quicksum(self.x[s,i,k1,k] for k1 in range(self.n_AGVs) for i in range(self.J) for s in range(self.clip(t-self.p[i]-2*self.delta+1),self.clip(t-self.p[i])) if not i in non_conflicting[j])
        self.model.addConstrs((starting_jobs(t,j,k)+conflicting_A_jobs(t,j,k)+conflicting_jobs(t,j,k) <= 1 for j in range(self.J) for t in range(self.T) for k in range(self.n_AGVs)), name='new')


        # Add Constants

        # The first A jobs are deterministic, for the second A jobs we have an earliest starting time. 
        for j in range(self.n_A_jobs):
            for t in range((j+1)*self.processing_times_A[0]+transportation):
                for k1, k2 in product(range(self.n_AGVs),repeat=2):
                    self.x_const[t,j,k1,k2] = 0

        # Two jobs from the same class are indistinguishable, we can therefore find a earliest starting time of the $n$-th job, which can be the $n$-th job scheduled.
        for j in range(self.n_A_jobs,self.n_A_jobs+self.n_B_jobs):
            j_B = j-self.n_A_jobs
            for t in range(self.delta+j_B*self.processing_times_B[0]):
                for k1, k2 in product(range(self.n_AGVs),repeat=2):
                    self.x_const[t,j,k1,k2] = 0

        for j in range(self.n_A_jobs + self.n_B_jobs, self.n_A_jobs + 2 * self.n_B_jobs):
            j_B = j - self.n_A_jobs - self.n_B_jobs
            for t in range(self.delta + j_B * self.processing_times_B[0] + transportation + self.processing_times_B[0]):
                for k1, k2 in product(range(self.n_AGVs), repeat=2):
                    self.x_const[(t, j, k1, k2)] = 0

        for j in range(self.n_A_jobs + 2 * self.n_B_jobs, self.n_A_jobs + 3 * self.n_B_jobs):
            j_B = j - self.n_A_jobs - self.n_B_jobs - self.n_B_jobs
            for t in range(self.delta + j_B * self.processing_times_B[0] + transportation + self.processing_times_B[0] + transportation + self.processing_times_B[1]):
                for k1, k2 in product(range(self.n_AGVs), repeat=2):
                    self.x_const[(t, j, k1, k2)] = 0
        
        for key, value in self.x_const.items():
            self.x[key].UB = value

        #self.y[0, 0].UB = 0

        n_x = self.T * self.J * self.n_AGVs * self.n_AGVs
        n_y = self.n_A_jobs * self.n_AGVs