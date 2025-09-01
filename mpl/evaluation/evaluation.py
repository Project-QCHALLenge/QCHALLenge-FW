import pandas as pd
import numpy as np

from mpl.models.mpl_gurobi_wait_overlap import MPLGurobiWaitOverlap
from abstract.evaluation.abstract_evaluation import AbstractEvaluation


class MPLEvaluation(AbstractEvaluation):
    def __init__(self, data, solution):
        self.data = data
        self.n_A_jobs = self.data.N_A
        self.n_B_jobs = self.data.N_B
        self.JOBS_A = list(range(self.data.N_A))
        self.JOBS_B = list(range(self.data.N_A, self.data.N_A + self.data.N_B))
        self.MACHINES = list(range(self.data.__data__["M"]))
        self.machine_names = self.data.machine_names
        self.TASKS = [0, 1]
        self.t_r = self.data.t_r
        self.T = self.data.__data__["T"]
        self.AGVS = list(range(self.data.__data__["R"]))

        self.solution_dict = solution
        if self.check_solution_type():
            self.solution = self.create_solution_df()
        else:
            self.solution = self.get_solution_reduced()
        self.solution_feasible = self.check_solution_feasible()

    def check_solution_type(self):
        for key in self.solution_dict:
            if key.startswith('x'):
                key_parts = key.split('_')
                return len(key_parts) == 6
        return False
    
    def get_solution_reduced(self):

        processing_times_A = self.data.processing_times["Jobs_A"]
        processing_times_B = self.data.processing_times["Jobs_B"]

        delta = self.data.t_r

        J = self.n_A_jobs + 3*self.n_B_jobs
        p = np.zeros(J)
        # processing time of A jobs
        p[:self.n_A_jobs] = processing_times_A[1]
        # processing time of B jobs
        self.n_A_jobs
        p[self.n_A_jobs:self.n_A_jobs+self.n_B_jobs] = processing_times_B[0]
        p[self.n_A_jobs+self.n_B_jobs:self.n_A_jobs+2*self.n_B_jobs] = processing_times_B[1]
        p[self.n_A_jobs+2*self.n_B_jobs:self.n_A_jobs+3*self.n_B_jobs] = processing_times_B[2]
        # make processing time integral for indexing
        p = p.astype(int)

        machines = [
            [j for j in range(self.n_A_jobs)] + 
            [j for j in range(self.n_A_jobs, self.n_A_jobs + self.n_B_jobs)] + 
            [j for j in range(self.n_A_jobs + 2 * self.n_B_jobs, self.n_A_jobs + 3 * self.n_B_jobs)],  # competition for machine 2
            [j for j in range(self.n_A_jobs + self.n_B_jobs, self.n_A_jobs + 2 * self.n_B_jobs)]
        ]

        starting_tuples = [(int(t), int(j), int(k1), int(k2)) 
                        for key, x_tjk1k2 in self.solution_dict.items() if key.startswith('x_') and x_tjk1k2 > 0.5
                        for _, t, j, k1, k2 in [key.split('_')]]
        
        starting_A_tuples = [(int(j), int(k)) 
                            for key, y_jk in self.solution_dict.items() if key.startswith('y_') and y_jk > 0.5
                            for _, j, k in [key.split('_')]]

        df_agv = pd.DataFrame({
            'time': [t-delta for (t,_,_,_) in starting_tuples ] + [int(t+p[j]) for (t,j,_,_) in starting_tuples] + [(j+1)*processing_times_A[0] for (j,_) in starting_A_tuples],
            'job': [j+self.n_A_jobs for (_,j,_,_) in starting_tuples] + [j+self.n_A_jobs for (_,j,_,_) in starting_tuples] + [j for (j,_) in starting_A_tuples],
            'agv': [k for (_,_,k,_) in starting_tuples] + [k for (_,_,_,k) in starting_tuples] + [k for (_,k) in starting_A_tuples],
            'type': ['to' for _ in starting_tuples ] + ['from' for _ in starting_tuples] + ['from' for _ in starting_A_tuples]
        })

        df_agv = df_agv.sort_values('time')
        agv_sparse = [[row[1]['agv'], row[1]['time'], row[1]['time']+delta-1, row[1]['job'], row[1]['type']] for  row in df_agv.iterrows()]

        schedule = {(n,0):n*processing_times_A[0] for n in range(self.n_A_jobs)}
        for key, x_tj in self.solution_dict.items():
            if key.startswith('x_') and x_tj > 0.5:
                _, t, j, k1, k2 = key.split('_')
                t, j = int(t), int(j)
                if j in machines[0]:
                    schedule[j+self.n_A_jobs,1]=t
                if j in machines[1]:
                    schedule[j+self.n_A_jobs,2]=t

        machine_names = {0:'M1_Sack-filling', 1:'M2_Stretch-hood', 2:'M3_Oktabin-filling'}
        df = pd.DataFrame()

        for key in schedule:
            j,m = key
            processing_times = p[j-self.n_A_jobs] if j >= self.n_A_jobs else processing_times_A[0]
            if j < 2*self.n_A_jobs:
                job_nr = j%self.n_A_jobs
                jobname = f'Job {job_nr}'
                job_type = f'A{job_nr}'
            else:
                job_nr = (j-2*self.n_A_jobs)%self.n_B_jobs
                jobname = f'Job {job_nr + self.n_A_jobs}'
                job_type = f'B{job_nr}'
            start_time = schedule[key]
            temp_df = pd.DataFrame([dict(
                JobName=jobname, 
                Process=f'Job {job_type}', 
                Start=int(start_time), 
                Finish=int(start_time+processing_times), 
                delta=int(processing_times), 
                Machines=machine_names[m])
            ])
            df = pd.concat([df, temp_df])

        # AGVs
        for agv, start, finish, j, from_to in agv_sparse:
            if j < 2*self.n_A_jobs:
                job_nr = j%self.n_A_jobs
                jobname = f'Job {job_nr}'
                job_type = f'A{job_nr}'
            else:
                job_nr = (j-2*self.n_A_jobs)%self.n_B_jobs
                jobname = f'Job {job_nr + self.n_A_jobs}'
                job_type = f'B{job_nr}'
            process = f"Keep-Job {job_type}" if from_to == "from" else f"Lift-Job {job_type}"
            temp_df = pd.DataFrame([dict(
                JobName=jobname, 
                Process=process, 
                Start=int(start), 
                Finish=int(finish+1), 
                delta=int(finish-start+1), 
                Machines=f"AGV_{agv}")
            ])
            df = pd.concat([df, temp_df])
        df = df.sort_values(by='Start')
        # df = self.clean_lift_jobs(df)
        return df
    
    def create_solution_df(self):
        schedule = self.extract_schedule()
        df_schedule_with_lift_overlap = self.schedule_to_df(schedule)
        # df_schedule = self.clean_lift_jobs(df_schedule_with_lift_overlap)
        return df_schedule_with_lift_overlap

    def extract_schedule(self):
        schedule = dict()
        for j in self.JOBS_A+self.JOBS_B:
            for m in self.MACHINES:
                for d in self.TASKS:
                    for t in range(self.T):
                        for r in self.AGVS:
                            key = f"x_{j}_{d}_{r}_{m}_{t}"
                            if key in self.solution_dict and self.solution_dict[key] == 1:
                                schedule[(j,m,d)]=(t,r)
        return schedule
    
    def schedule_to_df(self, schedule):
        df = pd.DataFrame()

        p_a = [
            self.data.processing_times["Jobs_A"][0],
            self.data.processing_times["Jobs_A"][1],
            0,
            0,
        ]
        p_b = [
            0,
            self.data.processing_times["Jobs_B"][0],
            self.data.processing_times["Jobs_B"][1],
            self.data.processing_times["Jobs_B"][2],
        ]

        p = [p_a] * len(self.JOBS_A)
        p.extend([p_b] * len(self.JOBS_B))

        for key in schedule:
            (j, m, d) = key
            if j < len(self.JOBS_A):
                job_type = f'A{j}'
            else:
                job_type = f'B{j - len(self.JOBS_A)}'
            if d == 0:
                start_time = schedule[key][0] - self.t_r
                temp_df = pd.DataFrame([dict(JobName=f'Job {j}', Process=f'Keep-Job {job_type}', Start=int(start_time),
                                             Finish=int(start_time + self.t_r), delta=int(self.t_r), Machines=f'AGV_{schedule[key][1]}')])
                df = pd.concat([df, temp_df])
            elif d == 1:
                if m == 0:
                    start_time = j * p[j][0]
                    job_type = f'A{j}'
                    temp_df = pd.DataFrame([dict(JobName=f'Job {j}', Process=f'Job {job_type}', Start=int(start_time),
                                                 Finish=int(start_time + p[j][0]), delta=int(p[j][0]),
                                                 Machines=self.machine_names[0])])
                    df = pd.concat([df, temp_df])
                start_time = schedule[key][0]
                temp_df = pd.DataFrame([dict(JobName=f'Job {j}', Process=f'Lift-Job {job_type}', Start=int(start_time),
                                             Finish=int(start_time + self.t_r), delta=int(self.t_r), Machines=f'AGV_{schedule[key][1]}')])
                df = pd.concat([df, temp_df])
            if d == 0:
                start_time = schedule[key][0]
                temp_df = pd.DataFrame([dict(JobName=f'Job {j}', Process=f'Job {job_type}', Start=int(start_time),
                                             Finish=int(start_time + p[j][m]), delta=int(p[j][m]), Machines=self.machine_names[m])])
                df = pd.concat([df, temp_df])
        df = df.sort_values(by='Start')
        return df
    
    def get_objective(self):
        return self.solution['Finish'].max()
    

    # def get_objective(self):
    #     u_keys_with_ones = [int(key.split('_')[1]) for key, value in self.solution_dict.items() if key.startswith('u_') and value == 1]
    #     if u_keys_with_ones:
    #         return max(u_keys_with_ones) + 1
    #     else:
    #         return -1
        
    def check_constraint(self, constraint):
        fulfilled = True
        return constraint, fulfilled

    # def check_constraint(self, constraint):
    #     fulfilled = True
    #     constraints_list = [constraint]
    #     submodel = MPLGurobiWaitOverlap(self.data)
    #     submodel.build_model_selected_constraints(constraints_list, self.prelim_schedule)
    #     answer = submodel.solve()
    #     sub_status = answer["status"]
    #     # new submodel consisting of constraint plus solution not feasible, therefore constraint not fulfilled
    #     if sub_status == 3:
    #         fulfilled = False

    #     return constraint, fulfilled


    def check_solution(self):
        constraints_list = list(range(1, 11))
        constraints_status = dict()
        for constraint in constraints_list:
            _, fulfilled = self.check_constraint(constraint)
            constraint_key = f"constraint {constraint}"
            if fulfilled:
                constraints_status[constraint_key] = []
            else:
                constraints_status[constraint_key] = [1]

        return constraints_status


    def get_solution(self):
        return self.solution

    def check_solution_feasible(self):
        constraint_status = self.check_solution()
        feasible = True
        for constraint, fulfilled in constraint_status.items():
            if len(fulfilled) > 0:
                return False
        return feasible
