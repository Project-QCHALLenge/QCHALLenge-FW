import time

import gurobipy as gp
import numpy as np

from itertools import product

from pas.data.pas_data import PASData

class GurobiConvexPAS:
    '''
    Production assignment scheduling as a routing problem 
    '''
    def __init__(self, data: PASData):
        self.data = data
        self.model, self.routing_variables, self.capacity_variables, self.processing_times_variables = self.build_model()
        self.model.Params.LogToConsole = 0
        self.model.Params.OutputFlag = 0

    @staticmethod
    def _create_eligible_jobs(
        eligible_machines: list,
        n_jobs: int,
        n_machines: int
    ) -> list[list[int]]:
        machine_to_job = [[] for _ in range(n_machines)]
        for job in range(n_jobs):
            for machine in eligible_machines[job]:
                machine_to_job[machine].append(job)
        return [list(set(jobs)) for jobs in machine_to_job]
    

    @staticmethod
    def _create_routing_variables(
        model : gp.Model,
        n_machines: int,
        eligible_jobs: list[list[int]]
    ) -> gp.tupledict[gp.Var]:
        keys = [(machine, job_1, job_2) for machine in range(n_machines) for job_1,job_2 in product([-1] + eligible_jobs[machine],repeat=2)]
        variables = {}
        for machine, job_1, job_2 in keys:
            variables[machine, job_1, job_2] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{machine}_{job_1}_{job_2}')

        return variables


    @staticmethod
    def _create_capacity_variables(
        model : gp.Model,
        n_machines: int,
        eligible_jobs: list[list[int]],
        processing_times: np.array
    ) -> gp.tupledict[gp.Var]:
        '''

        '''
        keys = [(machine, job) for machine in range(n_machines) for job in eligible_jobs[machine]]
        upper_bounds = {(machine, job): np.sum(processing_times[eligible_jobs[machine]]) for machine, job in keys}
        variables = {}
        for machine, job in keys:
            variables[machine, job] = model.addVar(
                vtype=gp.GRB.CONTINUOUS, 
                lb=0, 
                ub=upper_bounds[machine, job], 
                name=f'c_{machine}_{job}'
            )
        return variables
    

    @staticmethod
    def _create_processing_time_variables(
        model : gp.Model,
        n_machines: int,
        eligible_jobs: list[list[int]],
        processing_times: np.array
    ) -> gp.tupledict[gp.Var]:
        upper_bounds = [np.sum(processing_times[eligible_jobs[machine]]) for machine in range(n_machines)]
        variables = {}
        for m in range(n_machines):
            variables[(m)] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=upper_bounds[m], name=f'p_{m}')
        return variables


    @staticmethod
    def _constraint_every_job_exactly_once(
        model: gp.Model,
        routing_variables: gp.tupledict[gp.Var],
        eligible_machines: list[list[int]],
        eligible_jobs: list[list[int]],
        n_jobs: int
    ):
        '''
        A job must be processed by one eligible machine
        '''

        for job_1 in range(n_jobs):
            name = 'every_job_exactly_one_successor_' + str(job_1)
            model.addConstr(
                gp.quicksum(routing_variables[machine, job_1, job_2] for machine in eligible_machines[job_1] for job_2 in [-1] + eligible_jobs[machine]) == 1, name=name
            )

        for job_2 in range(n_jobs):
            name = 'every_job_exactly_one_predecessor_' + str(job_2)
            model.addConstr(
                gp.quicksum(routing_variables[machine, job_1, job_2] for machine in eligible_machines[job_2] for job_1 in [-1] + eligible_jobs[machine]) == 1, name=name
            )

    @staticmethod
    def _constraint_leave_dummy_job_at_most_once(
        model: gp.Model,
        routing_variables: gp.tupledict[gp.Var],
        n_machines: int,
        eligible_jobs: list[list[int]]
    ):
        '''
        
        '''
        for machine in range(n_machines):
            name = 'leave_dummy_' + str(machine)
            model.addConstr(
                gp.quicksum(routing_variables[machine, -1, job] for job in eligible_jobs[machine]) <= 1, name=name
            )

    
    @staticmethod
    def _constraint_flow_conservation(
        model: gp.Model,
        routing_variables: gp.tupledict[gp.Var],
        n_machines: int,
        eligible_jobs: list[list[int]]
    ):
        '''

        '''
        for machine in range(n_machines):
            for job in [-1] + eligible_jobs[machine]:
                name = 'flow_conservation_' + str(machine) + '_' + str(job)
                incoming = gp.quicksum(routing_variables[machine, predecessor, job] for predecessor in [-1] + eligible_jobs[machine])
                outgoing = gp.quicksum(routing_variables[machine, job, successor] for successor in [-1] + eligible_jobs[machine])
                model.addConstr(
                    incoming == outgoing, name=name
                )


    @staticmethod
    def _constraint_subtour_elimination(
        model: gp.Model,
        routing_variables: gp.tupledict[gp.Var],
        capacity_variables: gp.tupledict[gp.Var],
        n_machines: int, 
        eligible_jobs: list[list[int]],
        processing_times: np.array
    ):
        maximum_processing_times = [np.sum(processing_times[eligible_jobs[machine]]) for machine in range(n_machines)]
        for machine in range(n_machines):
            for job_1, job_2 in product(eligible_jobs[machine], repeat=2):
                model.addConstr(
                    capacity_variables[machine, job_1]-capacity_variables[machine, job_2] + maximum_processing_times[machine]*routing_variables[machine, job_1, job_2] <= maximum_processing_times[machine]-processing_times[job_2]
                )
        

    @staticmethod
    def _constraint_processing_times_coupling(
        model: gp.Model,
        capacity_variables: gp.tupledict[gp.Var],
        processing_times_variables: gp.tupledict[gp.Var],
        n_machines: int,
        eligible_jobs: list[list[int]]
    ):
        for machine in range(n_machines):
            for job in eligible_jobs[machine]:
                model.addConstr(
                    processing_times_variables[machine] >= capacity_variables[machine, job]
                )

    
    @staticmethod
    def _cutting_plane_sum_processing_times(
        model: gp.Model,
        processing_times_variables: gp.tupledict[gp.Var],
        processing_times: np.array
    ):
        model.addConstr(
            gp.quicksum(processing_times_variables.values()) == np.sum(processing_times)
        )


    @staticmethod
    def _objective_normalization(
        processing_times_variables: gp.tupledict[gp.Var],
    ) -> gp.QuadExpr:
        return gp.quicksum(variable**2 for variable in processing_times_variables.values())
    

    @staticmethod
    def _objective_normalization_approximation(
        model: gp.Model, 
        processing_times_variables: gp.tupledict[gp.Var],
        n_machines: int,
        total_processing_times: float
    ) -> gp.LinExpr:
        variable_maximum_processing_time = model.addVar(vtype=gp.GRB.CONTINUOUS)
        for machine in range(n_machines):
            model.addConstr(variable_maximum_processing_time >= processing_times_variables[machine])
        factor = int(total_processing_times/n_machines)
        return factor*variable_maximum_processing_time
    

    @staticmethod
    def _objective_setup_times(
        routing_variables: gp.tupledict[gp.Var],
        setup_times: np.array
    ) -> gp.LinExpr:
        return gp.quicksum(setup_times[job_1,job_2]*variable for (_, job_1, job_2), variable in routing_variables.items() if job_1 >= 0 and job_2 >= 0)
    

    @staticmethod
    def _objective_value(
        routing_variables: gp.tupledict[gp.Var],
        job_values: np.array
    ) -> gp.LinExpr:
        return gp.quicksum(-job_values[job,machine]*variable for (machine, _, job), variable in routing_variables.items() if job >= 0)
    

    def build_model(
        self,
        problem_typ: str = 'standard'
    ) -> tuple[gp.Model, gp.tupledict[gp.Var], gp.tupledict[gp.Var], gp.tupledict[gp.Var]]:
        n_jobs = self.data.j
        n_machines = self.data.m
        eligible_machines = self.data.eligible_machines
        eligible_jobs = self._create_eligible_jobs(eligible_machines, n_jobs, n_machines)

        setup_times = np.array(self.data.setup_times)
        processing_times = np.array(self.data.processing_times)
        job_values = np.array(self.data.job_values)

        model = gp.Model("Gurobi_convex")

        routing_variables = self._create_routing_variables(model, n_machines, eligible_jobs)
        capacity_variables = self._create_capacity_variables(model, n_machines, eligible_jobs, processing_times)
        processing_times_variables = self._create_processing_time_variables(model, n_machines, eligible_jobs, processing_times)
        self._constraint_every_job_exactly_once(model, routing_variables, eligible_machines, eligible_jobs, n_jobs)
        self._constraint_leave_dummy_job_at_most_once(model, routing_variables, n_machines, eligible_jobs)
        self._constraint_flow_conservation(model, routing_variables, n_machines, eligible_jobs)
        self._constraint_subtour_elimination(model, routing_variables, capacity_variables, n_machines, eligible_jobs, processing_times)
        self._constraint_processing_times_coupling(model, capacity_variables, processing_times_variables, n_machines, eligible_jobs)
        self._cutting_plane_sum_processing_times(model, processing_times_variables, processing_times)
        objective_normalization = self._objective_normalization(processing_times_variables) if problem_typ == 'standard' else self._objective_normalization_approximation(model, processing_times_variables, n_machines, np.sum(processing_times))
        objective_setup_times = self._objective_setup_times(routing_variables, setup_times)
        objective_value = self._objective_value(routing_variables, job_values)
        model.setObjective(objective_setup_times+objective_normalization+objective_value)

        model.update()

        return model, routing_variables, capacity_variables, processing_times_variables
    
    @classmethod
    def construct_sequences(
        cls,
        routing_variables: gp.tupledict[gp.Var]
    ):
        successors = {(machine, job_1): job_2 for (machine, job_1, job_2), variable in routing_variables.items() if variable.X > 0.5}
        sequences = {machine: [successors[machine,-1]] for machine, _ in successors.keys()}

        for machine in sequences.keys():
            while sequences[machine][-1] >= 0:
                job = sequences[machine][-1]
                sequences[machine].append(successors[machine, job])
            sequences[machine].pop()

        return sequences
    

    @staticmethod
    def add_start_solution(
        routing_variables:  gp.tupledict[gp.Var], 
        capacity_variables:  gp.tupledict[gp.Var], 
        processing_times_variables:  gp.tupledict[gp.Var],
        routing_variables_values: dict[int],
        capacity_variables_values: dict[int]
    ):
        for (machine, job_1, job_2), variable in routing_variables.items():
            variable.start = routing_variables_values[machine, job_1, job_2]

        for (machine, job), variable in capacity_variables.items():
            variable.start = capacity_variables_values[machine, job]

        processing_times_values = {machine: 0 for machine in processing_times_variables.keys()}
        for (machine, job), value in capacity_variables_values.items():
            processing_times_values[machine] = max(processing_times_values[machine], value)

        for machine, variable in processing_times_variables.items():
            variable.start = processing_times_values[machine]


    @staticmethod
    def add_partial_start_solution_from_routing(
        routing_variables: gp.tupledict[gp.Var],
        routing: dict[list[int]]
    ):
        for variable in routing_variables.values():
            variable.start = 0

        for machine, route in routing.items():
            routing_variables[machine, -1, route[0]].start = 1
            for t in range(1,len(route)):
                routing_variables[machine, route[t-1], route[t]].start = 1
            routing_variables[machine, route[-1], -1].start = 1

    def solve(self, **config):
        TimeLimit = config.get("TimeLimit", 1)
        self.model.Params.TimeLimit = TimeLimit

        start_time = time.time()
        self.model.optimize()
        runtime = time.time() - start_time

        sequences = self.convert_solution()
        _, solution_by_string = self.create_solution(sequences) 

        # solution = {var.VarName: int(var.X) for var in self.model.getVars()}

        self.solved = True

        return {"solution": solution_by_string, "energy": self.model.ObjVal, "runtime": runtime}
    
    def convert_solution(self):
        successors = {(machine, job_1): job_2 for (machine, job_1, job_2), variable in self.routing_variables.items() if variable.X > 0.5}
        sequences = {machine: [successors[machine,-1]] for machine, _ in successors.keys()}

        for machine in sequences.keys():
            while sequences[machine][-1] >= 0:
                job = sequences[machine][-1]
                sequences[machine].append(successors[machine, job])
            sequences[machine].pop()

        return sequences
    
    def create_solution(self, sequences):
        # initialize everything to 0
        solution_by_tuple = {(j,m,t): 0 for m in range(self.data.m) for j in range(self.data.j) for t in range(self.data.j)}
        # set variables to 1 if necessary
        for machine, sequence in sequences.items():
            for time, job in enumerate(sequence):
                solution_by_tuple[job, machine, time] = 1
        # also create a string
        name = lambda j,m,t : 'x_' + str(j) + '_' + str(m) + '_' + str(t)
        solution_by_string = {name(j,m,t) : value for (j,m,t), value in solution_by_tuple.items()}
        return solution_by_tuple, solution_by_string