# -*- coding: utf-8 -*-
from abstract.models.abstract_model import AbstractModel
import time
import dimod

import gurobipy as gp

from pathlib import Path
from gurobipy import GRB
from dwave.system import LeapHybridCQMSampler

from transformations import FromCQM

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class MPLCQMNoWaitOverlap(AbstractModel):
    def __init__(self, data, overlap = True):

        self.overlap = overlap
        
        self.TASKS = [0,1] # keeping (0), lifting (1)

        current_file_path = Path(__file__)
        self.tmp_output_path = current_file_path.parent / "model.lp"

        self.M = data.M
        self.machine_names = data.machine_names
        self.N_A = data.N_A
        self.N_B = data.N_B
        self.R = data.R
        self.T = data.T
        self.t_r = data.t_r
        #joba_time = ast.literal_eval(self.data.iloc[self.problem_index].joba_times) # type: ignore
        #jobb_time = ast.literal_eval(self.data.iloc[self.problem_index].jobb_times) # type: ignore
        self.processing_times = data.processing_times
        self.p_a = [self.processing_times['Jobs_A'][0],self.processing_times['Jobs_A'][1],0,0]
        self.p_b = [0,self.processing_times['Jobs_B'][0],self.processing_times['Jobs_B'][1],self.processing_times['Jobs_B'][2]]
        self.JOBS_A = list(range(self.N_A))
        self.JOBS_B = list(range(self.N_A,self.N_A+self.N_B))
        self.AGVS = list(range(self.R))
        self.MACHINES = list(range(self.M))
        self.total_machines = len(self.MACHINES)
        self.total_jobs = len(self.JOBS_A)+len(self.JOBS_B)

        self.p = [self.p_a] * len(self.JOBS_A)
        self.p.extend([self.p_b] * len(self.JOBS_B))

        self.processing_times_list = [self.p_a]*len(self.JOBS_A)
        self.processing_times_list.extend([self.p_b]*len(self.JOBS_B))

        self.model = self.build_model()

    
    def solve(self, **config):

        time_limit = config.get("timelimit", 1)

        sampler = LeapHybridCQMSampler()

        start_time = time.time()
        samples = sampler.sample_cqm(self.model, time_limit=time_limit)
        solve_time = time.time() - start_time

        feasible_sampleset = samples.filter(lambda row: row.is_feasible)
        if len(feasible_sampleset):      
            best_energy = feasible_sampleset.first.energy
            solutions_dict = feasible_sampleset.first.sample
        else:
            best_energy = samples.first.energy
            solutions_dict = samples.first.sample
        
        return {"solution": solutions_dict, "energy": best_energy, "runtime": solve_time}

    def solve_cqm(self, time_limit=None):
        
        sampler = LeapHybridCQMSampler()
        samples = sampler.sample_cqm(self.model, time_limit=time_limit)

        feasible_sampleset = samples.filter(lambda row: row.is_feasible)  
        best_sample=None
        best_energy = None
        self.solutions_dict = dict()
        if len(feasible_sampleset):      
            best_sample = feasible_sampleset.first
            print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(samples)))
        
            best_energy = feasible_sampleset.first.energy
            self.solutions_dict = feasible_sampleset.first.sample
        
        return best_energy, best_sample, feasible_sampleset, samples
    

    def extract_schedule(self):
        x_vals=dict()
        schedule = dict()
        if len(self.solutions_dict)>0:
            for j in self.JOBS_A+self.JOBS_B:
                for d in self.TASKS:
                    for r in self.AGVS:
                        for m in self.MACHINES:
                            for t in range(self.T):
                                if f'x_{j}_{d}_{r}_{m}_{t}' in self.solutions_dict:
                                    x_vals[(j,d,r,m,t)] = self.solutions_dict[f'x_{j}_{d}_{r}_{m}_{t}']
                                elif (j,d,r,m,t) in self.constants.keys():
                                    x_vals[(j,d,r,m,t)] = self.constants[(j,d,r,m,t)]
                                else:
                                    x_vals[(j,d,r,m,t)]=0
                                if x_vals[(j,d,r,m,t)]==1:
                                    schedule[(j,m,d)]=(t,r)

        return x_vals, schedule

    def build_model(self):
        cqm, total_bits = self.build_gurobi_model()
        return cqm

    def build_gurobi_model(self):
        model = gp.Model("miqp")
        x = dict()
        fix = []
        constants=dict()
        # No B jobs need to be kept/lifted on machine M1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(self.T):
                        fix.append((f'x_{j}_{d}_{r}_{0}_{t}',0))
                        constants[(j,d,r,0,t)] = 0
                        x[(j,d,r,0,t)] = 0
        ######### DONE ###########
        # A jobs need not be kept on machine M1 ( they are assumed to be kept automatically)
        for j in self.JOBS_A:
            for r in self.AGVS:
                for t in range(self.T):
                    fix.append((f'x_{j}_{0}_{r}_{0}_{t}',0))
                    constants[(j,0,r,0,t)] = 0
                    x[(j,0,r,0,t)] = 0
        ######### DONE ###########
        # The A jobs on machine M1 cannot be lifted before they are processed on M1
        for j in self.JOBS_A:
            consumed_time = (j+1)*self.processing_times['Jobs_A'][0]
            for r in self.AGVS:
                for t in range(0,consumed_time):
                    fix.append((f'x_{j}_{1}_{r}_{0}_{t}',0))
                    constants[(j,1,r,0,t)] = 0
                    x[(j,1,r,0,t)] = 0
        ######### DONE ###########
        # The A jobs on machine M1 cannot be lifted after the processing of the next job on M1
        for j in self.JOBS_A:
            final_lifting_time = (j+2)*self.processing_times['Jobs_A'][0]
            for r in self.AGVS:
                for t in range(final_lifting_time,self.T):
                    fix.append((f'x_{j}_{1}_{r}_{0}_{t}',0))
                    constants[(j,1,r,0,t)] = 0
                    x[(j,1,r,0,t)] = 0
        ######### DONE ###########
        # The A jobs on machine M2 cannot be kept before they are processed on M1 and transported to M2, and cannot be lifted before its minimum finishing time on M2
        for j in self.JOBS_A:
            for d in self.TASKS:
                consumed_time = (j+1)*self.processing_times['Jobs_A'][0]+self.t_r + d*(self.processing_times['Jobs_A'][1])
                for r in self.AGVS:
                    for t in range(0,consumed_time):
                        fix.append((f'x_{j}_{d}_{r}_{1}_{t}',0))
                        constants[(j,d,r,1,t)] = 0
                        x[(j,d,r,1,t)] = 0
        ######### DONE ###########
        # no A jobs need to be processed on M3 and M4
        for j in self.JOBS_A:
            for d in self.TASKS:
                for r in self.AGVS:
                    for m in [2,3]:
                        for t in range(self.T):
                            fix.append((f'x_{j}_{d}_{r}_{m}_{t}',0))
                            constants[(j,d,r,m,t)] = 0  
                            x[(j,d,r,m,t)] = 0  
        ######### DONE ###########
        # no B jobs can be kept on machine M2 from time t=0 to t=t_r-1, and lifted from t=0 to t=t_r+p_j^m-1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(self.t_r + d*(self.processing_times['Jobs_B'][0])):
                        fix.append((f'x_{j}_{d}_{r}_{1}_{t}',0))
                        constants[(j,d,r,1,t)] = 0
                        x[(j,d,r,1,t)] = 0
        ######### DONE ###########
        # no job B can be kept on machine M3 from t=0 and t=t_r+p_j^m+r_t-1 and lifted from M3 between time t=0 to t=t_r+p_b^0+t_r-1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(self.t_r + self.processing_times['Jobs_B'][0] + self.t_r + d*(self.processing_times['Jobs_B'][1])):
                        fix.append((f'x_{j}_{d}_{r}_{2}_{t}',0))
                        constants[(j,d,r,2,t)] = 0
                        x[(j,d,r,2,t)] = 0
        
        ######### DONE ###########
        # no job B can be kept on machine M4 from t=0 and t=t_r+p_j^m+r_t-1 and lifted from M3 between time t=0 to t=t_r+p_b^0+t_r-1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(self.t_r + self.processing_times['Jobs_B'][0] + self.t_r + self.processing_times['Jobs_B'][1] + self.t_r + d*(self.processing_times['Jobs_B'][2])):
                        fix.append((f'x_{j}_{d}_{r}_{3}_{t}',0))
                        constants[(j,d,r,3,t)] = 0
                        x[(j,d,r,3,t)] = 0
        ######### DONE ###########
        ### if a job has a processing time of p on machine m, then it cant be kept from T-p to T

        for j in self.JOBS_B:
            for m in self.MACHINES:
                for r in self.AGVS:
                    for t in range(self.T-self.p_b[m]-1,self.T):
                        fix.append((f'x_{j}_{0}_{r}_{m}_{t}',0))
                        constants[(j,0,r,m,t)] = 0
                        x[(j,0,r,m,t)] = 0
        ######### DONE ###########
        for j in self.JOBS_A:
            for m in self.MACHINES:
                for r in self.AGVS:
                    for t in range(self.T-self.p_a[m]-1,self.T):
                        fix.append((f'x_{j}_{0}_{r}_{m}_{t}',0))
                        constants[(j,0,r,m,t)] = 0
                        x[(j,0,r,m,t)] = 0
        ######### DONE ###########
        # no A job can be lifted from M1 on any time other than J*p_a[0]
        for j in self.JOBS_A:
            for r in self.AGVS:
                for t in range(self.T):
                    if t!=(j+1)*self.p_a[0]:
                        fix.append((f'x_{j}_{1}_{r}_{0}_{t}',0))
                        x[(j,1,r,0,t)] = 0
                        constants[(j,1,r,0,t)] = 0
        ######### DONE ###########
        # the first A job must be lifted by AGV 0 at time p_a[0]
        # fix.append((f'x_{0}_{1}_{0}_{0}_{self.p_a[0]}',1))
        # constants[(0,1,0,0, self.p_a[0] )] = 1
        # x[(0,1,0,0, self.p_a[0] )] = 1
        # for r in self.AGVS:
        #     if r!=0:
        #         fix.append((f'x_{0}_{1}_{r}_{0}_{self.p_a[0]}',0))
        #         constants[(0,1,r,0, self.p_a[0] )] = 0
        #         x[(0,1,r,0, self.p_a[0] )] = 0
        ######### DONE ###########
                        

        for j in self.JOBS_A+self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for m in self.MACHINES:
                        for t in range(self.T):
                            if (j,d,r,m,t) not in constants:
                                x[(j,d,r,m,t)] = model.addVar(lb=0.0, ub=1.0, name=f'x_{j}_{d}_{r}_{m}_{t}')
                                x[(j,d,r,m,t)].VType = GRB.BINARY

        # C_max = model.addVar(obj=1, lb=self.p_a[1]*len(self.JOBS_A) + self.p_b[1]*len(self.JOBS_B) + self.p_b[3]*len(self.JOBS_B))
        u = dict()
        for t in range(self.T):
            u[t] = model.addVar(lb=0.0, ub=1.0, name=f'u_{t}')
            u[t].VType = GRB.BINARY
        # Constraint 1
        for j in self.JOBS_A:
            model.addLConstr(sum([x[(j,1,r,0, (j+1)*self.p_a[0] )] for r in self.AGVS]) == 1)  # type: ignore

        # Constraint 2
        for r in self.AGVS:
            for t in range(self.T):    
                model.addLConstr(sum([x[(j,d,r,m,t)] for j in self.JOBS_A+self.JOBS_B for d in self.TASKS for m in self.MACHINES]) <= 1)  # type: ignore

        # Constraint 3.1
        for j in self.JOBS_A+self.JOBS_B:
            for d in self.TASKS:
                model.addLConstr(sum([x[(j,d,r,1,t)] for t in range(self.t_r, self.T) for r in self.AGVS]) == 1) # type: ignore

        # Constraint 3.2
        for j in self.JOBS_A:
            for t in range(self.t_r, self.T):
                first_term = sum([x[(j,0,r,1,t)] for r in self.AGVS ])
                second_term = sum([x[(j,1,r,1,min(t+self.p_a[1], self.T-1))] for r in self.AGVS])
                model.addLConstr(first_term - second_term == 0)  # type: ignore

        # Constraint 3.3
        for j in self.JOBS_B:
            for t in range(self.T):
                first_term = sum([x[(j,0,r,1,t)] for r in self.AGVS ])
                second_term = sum([x[(j,1,r,1,min(t+self.p_b[1], self.T-1))] for r in self.AGVS])
                model.addLConstr(first_term - second_term == 0)  # type: ignore

        # Constraint 4.1
        for j in self.JOBS_B:
            for d in self.TASKS:
                model.addLConstr(sum([x[(j,d,r,2,t)] for t in range(2*self.t_r+self.p_b[1],self.T) for r in self.AGVS]) == 1)  # type: ignore

        # Constraint 4.2
        for j in self.JOBS_B:
            for t in range(2*self.t_r+self.p_b[1],self.T):
                first_term = sum([x[(j,0,r,2,t)] for r in self.AGVS ])
                second_term = sum([x[(j,1,r,2,min(t+self.p_b[2],self.T-1))] for r in self.AGVS])
                model.addLConstr(first_term - second_term == 0)  # type: ignore

        # Constraint 5.1
        for j in self.JOBS_B:
            for d in self.TASKS:
                model.addLConstr(sum([x[(j,d,r,3,t)] for t in range(3*self.t_r+self.p_b[1]+self.p_b[2],self.T) for r in self.AGVS]) == 1)  # type: ignore

        # Constraint 5.2
        for j in self.JOBS_B:
            for t in range(3*self.t_r+self.p_b[1]+self.p_b[2],self.T):
                first_term = sum([x[(j,0,r,3,t)] for r in self.AGVS ])
                second_term = sum([x[(j,1,r,3,min(t+self.p_b[3],self.T-1))] for r in self.AGVS])
                model.addLConstr(first_term - second_term == 0)  # type: ignore

        # Constraint 6
        for m in [0,1,2]:
            for j in self.JOBS_A+self.JOBS_B:
                for r in self.AGVS:
                    full_term = 0
                    for t in range(self.T):
                        start_time = max(t-self.t_r+1, 0)
                        first_term = x[(j,0,r,m+1,t)]
                        second_term = sum([x[(j,1,r1,m,t1)] for t1 in range(start_time,self.T) for r1 in self.AGVS])
                        full_term += first_term * second_term
                    model.addQConstr(full_term == 0)

        # Constraint 7.1
        for j1 in self.JOBS_A+self.JOBS_B:
            for m in [0,2]:
                for r1 in self.AGVS:
                    full_term = 0
                    for t1 in range(self.T):
                        if j1<=self.N_A-1:
                            p = self.p_a[m]
                        else:
                            p = self.p_b[m]
                        end_time = min(t1+p, self.T-1)
                        first_term = x[(j1,0,r1,m,t1)]
                        second_term = sum([x[(j2,0,r2,m,t2)] for j2 in self.JOBS_A+self.JOBS_B if j1!=j2 for t2 in range(t1,end_time) for r2 in self.AGVS])
                        full_term += first_term * second_term
                    model.addQConstr(full_term == 0)
                    
                        
        # Constraint 7.2
        for j1 in self.JOBS_A+self.JOBS_B:
            for m1 in [1,3]:
                for r1 in self.AGVS:
                    full_term = 0
                    for t1 in range(self.T):
                        end_time = min(t1+p, self.T-1)
                        if j1<=self.N_A-1:
                            p = self.p_a[m1]
                        else:
                            p = self.p_b[m1]
                        first_term = x[(j1,0,r1,m1,t1)]
                        second_term = sum([x[(j2,0,r2,m2,t2)]  for j2 in self.JOBS_A+self.JOBS_B if j1!=j2 for t2 in range(t1,end_time) for m2 in [1,3] for r2 in self.AGVS])
                        full_term += first_term * second_term
                    model.addQConstr(full_term == 0)
                       
                        
                
        # Constraint 8.1
        for r in self.AGVS:
            for m in [0,1,2,3]:
                for i in self.JOBS_A+self.JOBS_B:
                    full_term = 0
                    for t in range(self.T):
                        first_term = x[(i,0,r,m,t)]
                        second_term = sum([x[(j,0,r,m1,t+bot_time)]  for m1 in [0,1,2,3] for j in self.JOBS_A+self.JOBS_B if i!=j for bot_time in range(1,2*self.t_r) if t+bot_time<=self.T-1])
                        full_term += first_term * second_term
                    model.addQConstr(full_term == 0)
                    
                        
        # Constraint 8.2
        for r in self.AGVS:
            for m in [0,1,2,3]:
                for i in self.JOBS_A+self.JOBS_B:
                    full_term = 0
                    for t in range(self.T):
                        first_term = x[(i,1,r,m,t)]
                        second_term = sum([x[(j,1,r,m1,t+bot_time)] for m1 in [0,1,2,3] for j in self.JOBS_A+self.JOBS_B if i!=j for bot_time in range(2*self.t_r) if t+bot_time<=self.T-1 ])
                        full_term += first_term * second_term
                    model.addQConstr(full_term == 0)
                    

        # Constraint 8.3.1
        for r in self.AGVS:
            for m in [0,1,2,3]:
                for j in self.JOBS_A+self.JOBS_B:
                    full_term = 0
                    for t in range(self.T):
                        first_term = x[(j,1,r,m,t)]
                        second_term = sum([x[(j,0,r,m1,t+bot_time)] for m1 in [0,1,2,3] if m!=m1 for bot_time in range(self.t_r) if t+bot_time<=self.T-1 ])
                        full_term += first_term * second_term
                    model.addQConstr(full_term == 0)
                    
        # Constraint 8.3.2
        for r in self.AGVS:
            for m in [0,1,2,3]:
                for j in self.JOBS_A+self.JOBS_B:
                    full_term = 0
                    for t in range(self.T):
                        first_term = x[(j,1,r,m,t)]
                        second_term = sum([x[(j1,0,r,m1,t+bot_time)] for m1 in [0,1,2,3] for j1 in self.JOBS_A+self.JOBS_B if j!=j1 for bot_time in range(2*self.t_r) if t+bot_time<=self.T-1 ])
                        full_term += first_term * second_term
                    model.addQConstr( full_term == 0)
                    

        # Constraint 8.3.3
        for r in self.AGVS:
            for m in [0,1,2,3]:
                for j in self.JOBS_A+self.JOBS_B:
                    full_term = 0
                    for t in range(self.T):
                        first_term =  x[(j,0,r,m,t)] 
                        second_term = sum([x[(j1,1,r,m1,t+bot_time)] for m1 in [0,1,2,3] for j1 in self.JOBS_A+self.JOBS_B if j!=j1 for bot_time in range(self.t_r) if t+bot_time<=self.T-1 ])
                        full_term += first_term * second_term
                    model.addQConstr( full_term == 0)
                    

        
        # Constraint 8.4
        if self.overlap:
            for j in self.JOBS_A+self.JOBS_B:
                for r1 in self.AGVS:
                    for m in [0,1,2,3]:
                        full_term = 0
                        for t in range(self.T):
                            first_term = x[(j,1,r1,m,t)]
                            second_term = sum([ x[(j,0,r2,m1,t+bot_time)] for r2 in self.AGVS if r1!=r2 for m1 in [0,1,2,3] for bot_time in range(2*self.t_r) if t+bot_time<=self.T-1 ])
                            full_term += first_term * second_term
                        model.addQConstr( full_term == 0)
        else:
            for j in self.JOBS_A+self.JOBS_B:
                for r1 in self.AGVS:
                    for m in [0,1,2,3]:
                        full_term = 0
                        for t in range(self.T):
                            first_term = x[(j,1,r1,m,t)]
                            second_term = sum([ x[(j,0,r2,m1,t+bot_time)] for r2 in self.AGVS for m1 in [0,1,2,3] for bot_time in range(2*self.t_r) if t+bot_time<=self.T-1 ])
                            full_term += first_term * second_term
                        model.addQConstr( full_term == 0)


        

        # model.addConstrs(
        # gp.quicksum((t+self.p_a[m])*x[(j,0,r,m,t)] for t in range(self.T) for r in self.AGVS ) <= C_max for j in self.JOBS_A for m in [1])
        # model.addConstrs(
        # gp.quicksum((t+self.p_b[m])*x[(j,0,r,m,t)] for t in range(self.T) for r in self.AGVS ) <= C_max for j in self.JOBS_B for m in [3])

        model.addLConstr(sum([u[t] for t in range(self.T)]) == 1) 

        for j in self.JOBS_A+self.JOBS_B:
            for r in self.AGVS:
                for m in [0,1,2,3]:
                    full_term=0
                    for t1 in range(1,self.T):
                        for t2 in range(t1+1):
                            full_term += x[(j,1,r,m,t1)]*u[t2]
                    model.addQConstr(full_term  == 0)
        H=sum([t*u[t] for t in range(self.T) ])
        model.setObjective(H,GRB.MINIMIZE)

        model.write(self.tmp_output_path.as_posix())
        self.fix=fix
        self.constants=constants
        cqm = dimod.lp.load(self.tmp_output_path.as_posix()) 
        cqm=cqm.fix_variables([x for x in self.fix if x[0] in cqm.variables],inplace=False)
        total_bits = len(cqm.variables)

        return cqm, total_bits

    #############################################################################################
    #############################################################################################
    def load_cqm_model(self):
        fix = []
        constants=dict()
        # No B jobs need to be kept/lifted on machine M1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(self.T):
                        fix.append((f'x_{j}_{d}_{r}_{0}_{t}',0))
                        constants[(j,d,r,0,t)] = 0
        ######### DONE ###########
        # A jobs need not be kept on machine M1 ( they are assumed to be kept automatically)
        for j in self.JOBS_A:
            for r in self.AGVS:
                for t in range(self.T):
                    fix.append((f'x_{j}_{0}_{r}_{0}_{t}',0))
                    constants[(j,0,r,0,t)] = 0
        ######### DONE ###########
        # The A jobs on machine M1 cannot be lifted before they are processed on M1
        for j in self.JOBS_A:
            consumed_time = (j+1)*self.processing_times['Jobs_A'][0]
            for r in self.AGVS:
                for t in range(0,consumed_time):
                    fix.append((f'x_{j}_{1}_{r}_{0}_{t}',0))
                    constants[(j,1,r,0,t)] = 0
        ######### DONE ###########
        # The A jobs on machine M1 cannot be lifted after the processing of the next job on M1
        for j in self.JOBS_A:
            final_lifting_time = (j+2)*self.processing_times['Jobs_A'][0]
            for r in self.AGVS:
                for t in range(final_lifting_time,self.T):
                    fix.append((f'x_{j}_{1}_{r}_{0}_{t}',0))
                    constants[(j,1,r,0,t)] = 0
        ######### DONE ###########
        # The A jobs on machine M2 cannot be kept before they are processed on M1 and transported to M2, and cannot be lifted before its minimum finishing time on M2
        for j in self.JOBS_A:
            for d in self.TASKS:
                consumed_time = (j+1)*self.processing_times['Jobs_A'][0]+self.t_r + d*(self.processing_times['Jobs_A'][1])
                for r in self.AGVS:
                    for t in range(0,consumed_time):
                        fix.append((f'x_{j}_{d}_{r}_{1}_{t}',0))
                        constants[(j,d,r,1,t)] = 0
        ######### DONE ###########
        # no A jobs need to be processed on M3 and M4
        for j in self.JOBS_A:
            for d in self.TASKS:
                for r in self.AGVS:
                    for m in [2,3]:
                        for t in range(self.T):
                            fix.append((f'x_{j}_{d}_{r}_{m}_{t}',0))
                            constants[(j,d,r,m,t)] = 0  
        ######### DONE ###########
        # no B jobs can be kept on machine M2 from time t=0 to t=t_r-1, and lifted from t=0 to t=t_r+p_j^m-1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(self.t_r + d*(self.processing_times['Jobs_B'][0])):
                        fix.append((f'x_{j}_{d}_{r}_{1}_{t}',0))
                        constants[(j,d,r,1,t)] = 0
        ######### DONE ###########
        # no job B can be kept on machine M3 from t=0 and t=t_r+p_j^m+r_t-1 and lifted from M3 between time t=0 to t=t_r+p_b^0+t_r-1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(self.t_r + self.processing_times['Jobs_B'][0] + self.t_r + d*(self.processing_times['Jobs_B'][1])):
                        fix.append((f'x_{j}_{d}_{r}_{2}_{t}',0))
                        constants[(j,d,r,2,t)] = 0
        
        ######### DONE ###########
        # no job B can be kept on machine M4 from t=0 and t=t_r+p_j^m+r_t-1 and lifted from M3 between time t=0 to t=t_r+p_b^0+t_r-1
        for j in self.JOBS_B:
            for d in self.TASKS:
                for r in self.AGVS:
                    for t in range(self.t_r + self.processing_times['Jobs_B'][0] + self.t_r + self.processing_times['Jobs_B'][1] + self.t_r + d*(self.processing_times['Jobs_B'][2])):
                        fix.append((f'x_{j}_{d}_{r}_{3}_{t}',0))
                        constants[(j,d,r,3,t)] = 0
        ######### DONE ###########
        ### if a job has a processing time of p on machine m, then it cant be kept from T-p to T

        for j in self.JOBS_B:
            for m in self.MACHINES:
                for r in self.AGVS:
                    for t in range(self.T-self.p_b[m]-1,self.T):
                        fix.append((f'x_{j}_{0}_{r}_{m}_{t}',0))
                        constants[(j,0,r,m,t)] = 0
        ######### DONE ###########
        for j in self.JOBS_A:
            for m in self.MACHINES:
                for r in self.AGVS:
                    for t in range(self.T-self.p_a[m]-1,self.T):
                        fix.append((f'x_{j}_{0}_{r}_{m}_{t}',0))
                        constants[(j,0,r,m,t)] = 0
        ######### DONE ###########
        # no A job can be lifted from M1 on any time other than J*p_a[0]
        for j in self.JOBS_A:
            for r in self.AGVS:
                for t in range(self.T):
                    if t!=(j+1)*self.p_a[0]:
                        fix.append((f'x_{j}_{1}_{r}_{0}_{t}',0))
                        constants[(j,1,r,0,t)] = 0
        ######### DONE ###########
        # the first A job must be lifted by AGV 0 at time p_a[0]
        # fix.append((f'x_{0}_{1}_{0}_{0}_{self.p_a[0]}',1))
        # constants[(0,1,0,0, self.p_a[0] )] = 1
        # x[(0,1,0,0, self.p_a[0] )] = 1
        # for r in self.AGVS:
        #     if r!=0:
        #         fix.append((f'x_{0}_{1}_{r}_{0}_{self.p_a[0]}',0))
        #         constants[(0,1,r,0, self.p_a[0] )] = 0
        #         x[(0,1,r,0, self.p_a[0] )] = 0
        ######### DONE ########### 

        self.fix=fix
        self.constants=constants
        cqm = dimod.lp.load(self.tmp_output_path.as_posix()) 
        cqm=cqm.fix_variables([x for x in self.fix if x[0] in cqm.variables],inplace=False)
        print("Model loaded")
        total_bits = len(cqm.variables)

        return cqm, total_bits
    
    ############################################################################################
    