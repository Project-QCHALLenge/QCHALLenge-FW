import time
from abstract.models.abstract_model import AbstractModel
from enum import Enum
from itertools import combinations
from dimod import ConstrainedQuadraticModel, Binary, Integer
from docplex.mp.model import Model as CplexModel
from docplex.mp.constants import ComparisonType, ObjectiveSense

from pas.data.pas_data import PASData
from transformations import FromCPLEX


class Sense(Enum):
    Min = "min"
    Max = "max"


class CplexPAS(AbstractModel):
    """
    Class to create a Cplex model for the product assignment problem
    """

    def __init__(self, data: PASData):
        self.data: PASData = data
        self.__variable_keys = self.allowed_variables()
        self.__model = CplexModel(name="PAS")
        self.__variables = self.__model.binary_var_dict(keys=self.__variable_keys, name="x")
        self.__build_model()


    def solve(self, **config):
        TimeLimit = config.get("TimeLimit", 1)
        self.__model.set_time_limit(TimeLimit)

        start_time = time.time()
        self.__model.solve()
        runtime = time.time() - start_time  

        solution = {var.name: var.solution_value for var in self.__model.iter_variables()}

        return {"solution": solution, "runtime": runtime}
    
    def solve_qubo(self, solve_func, **config):

        qubo, converter = FromCPLEX(self.__model).to_matrix()

        start_time = time.time()
        answer = solve_func(Q=qubo, **config)
        runtime = time.time() - start_time  

        solution = converter(answer.first.sample)
        
        return {"solution": solution, "energy": answer.first.energy, "runtime": runtime}

    def from_cplex_to_cqm(self):
        """
        Convert the model to a CQML model
        """
        cqm = ConstrainedQuadraticModel()

        vardict = {}
        for var in self.__model.iter_variables():

            if var.is_binary():
                vardict[var.name] = Binary(var.name)

            elif var.is_integer():
                vardict[var.name] = Integer(var.name)

            else:
                raise RuntimeError(
                    f"variables of type '{var.vartype.cplex_typecode}’ is not supported"
                )


        # Translate the constraints
        for constr in self.__model.iter_constraints():
            left = constr.left_expr
            right = constr.right_expr

            lhs = (
                    sum(vardict[var.name] * bias for var, bias in left.iter_terms())
                    + left.get_constant()
            )
            lhs += sum(
                vardict[var1.name] * vardict[var2.name] * bias
                for var1, var2, bias in left.iter_quad_triplets()
            )

            rhs = (
                    sum(vardict[var.name] * bias for var, bias in right.iter_terms())
                    + right.get_constant()
            )
            rhs += sum(
                vardict[var1.name] * vardict[var2.name] * bias
                for var1, var2, bias in right.iter_quad_triplets()
            )

            if constr.sense == ComparisonType.EQ:
                cqm.add_constraint(lhs - rhs == 0)
            elif constr.sense == ComparisonType.LE:
                cqm.add_constraint(lhs - rhs <= 0)
            elif constr.sense == ComparisonType.GE:
                cqm.add_constraint(lhs - rhs >= 0)

        # Translate the objective
        obj_expr = self.__model.get_objective_expr()
        obj = sum(vardict[var.name] * bias for var, bias in obj_expr.iter_terms())
        obj += sum(
            vardict[var1.name] * vardict[var2.name] * bias
            for var1, var2, bias in obj_expr.iter_quad_triplets()
        )
        if self.__model.objective_sense == ObjectiveSense.Maximize:
            obj *= -1
        elif self.__model.objective_sense == ObjectiveSense.Minimize:
            pass

        obj += obj_expr.get_constant()
        cqm.set_objective(obj)
        return cqm

    def build_model(self, *args, **kwargs):
        self.__build_model()

    def __build_model(self):
        """
        Build the model
        """
        # Add the 3 objective functions
        job_value = self.job_value_expression()
        setup_times = self.setup_times_expression()
        normalize_times = self.normalize_times_expression()
        objective_expr = (- job_value + setup_times +
                          normalize_times)
        self.__model.set_objective(
            sense=Sense.Min.value,
            expr=objective_expr
        )

        # Add the constraints
        self.c3_max_one_job_at_a_time()
        self.c4_no_timesteps_skipped()
        self.c5_all_jobs_done_once()

    def job_value_expression(self):
        expr = sum(self.data.job_values[j][m]*var
                   for (j, m, _), var in
                   self.__variables.items())
        return expr

    def setup_times_expression(self):
        """
        Set the objective function to minimize the sum of the setup times
        """
        expr = 0
        for j1, j2 in combinations(range(self.data.j), 2):
            shared_machines = set(self.data.eligible_machines[j1]).intersection(set(self.data.eligible_machines[j2]))
            for m in shared_machines:
                for n in range(self.data.jobs_per_machine[m] - 1):
                    expr += self.data.setup_times[j1][j2]*self.__variables[(j1, m, n)]*self.__variables[j2, m, n+1]
                    expr += self.data.setup_times[j2][j1]*self.__variables[(j2, m, n)]*self.__variables[j1, m, n+1]

        return expr

    def normalize_times_expression(self):
        """
        Set the objective function to minimize the sum of the setup times
        """
        expr = 0
        for m in range(self.data.m):
            expr += sum(
                self.data.processing_times[j]* self.__variables[j, m, n]
                for j in self.data.eligible_jobs[m] for n in range(self.data.jobs_per_machine[m])
                )**2
        return expr

    def c3_max_one_job_at_a_time(self):
        """
        Set the constraint that at most one job can be processed at a time
        """
        for m in range(self.data.m):
            for n in range(self.data.jobs_per_machine[m]):
                self.__model.add_constraint(
                    ct=sum(self.__variables[j, m, n] for j in self.data.eligible_jobs[m]) <= 1
                )

    def c4_no_timesteps_skipped(self):
        """
        Set the constraint that no timesteps can be skipped
        """
        for m in range(self.data.m):
            eligible_jobs = self.data.eligible_jobs[m]
            for n in range(self.data.jobs_per_machine[m] - 1):
                self.__model.add_constraint(
                    ct=sum(self.__variables[j, m, n+1] for j in eligible_jobs) <= sum(self.__variables[j, m, n] for j in eligible_jobs)
                )

    def c5_all_jobs_done_once(self):
        """
        Set the constraint that all jobs must be done once
        """
        for j in range(self.data.j):
            self.__model.add_constraint(
                ct=sum(self.__variables[j, m, n] for m in self.data.eligible_machines[j] for n in range(self.data.jobs_per_machine[m])) == 1
            )

    def allowed_variables(self):
        allowed_variables = [
            (j, m, n) for j in range(self.data.j)
            for m in self.data.eligible_machines[j]
            for n in range(self.data.jobs_per_machine[m])
        ]
        return allowed_variables

    @property
    def variables(self):
        return self.__variables

    @property
    def model(self):
        return self.__model









