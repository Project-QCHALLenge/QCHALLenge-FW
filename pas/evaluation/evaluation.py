import numpy.typing as npt

from typing import Union
from docplex.mp.solution import SolveSolution
from numpy import ndarray
from re import findall

from pas.data.pas_data import PASData
from pas.data.utils import SolutionType, ErrorDict
from abstract.evaluation.abstract_evaluation import AbstractEvaluation


class EvaluationPAS(AbstractEvaluation):
    """
    Evaluation of a solution to the PAS problem.
    """

    def __init__(self, data: PASData, solution: Union[list[int], dict[str, int], npt.NDArray, SolutionType], convert_solution: bool = True):
        self.data = data
        if convert_solution:
            self.solution = self.set_solution(solution)
        else:
            self.solution = solution

    def set_solution(self, solution: Union[list[int], dict[str, int], npt.NDArray]
                     ):
        """
        Set the solution to evaluate accordingly to the format if it comes from gurobi/cplex or qubo model.
        """
        if isinstance(solution, dict):
            return self.decode_solution_dict(solution)

        elif isinstance(solution, list) or isinstance(solution, ndarray):
            return self.decode_solution_array(solution)

        elif isinstance(solution, SolveSolution):
            solution_dict = solution.as_name_dict()
            return self.decode_solution_dict(solution_dict)

        else:
            raise ValueError("The solution must be a dictionary ({x_j_m_n:0/1}), list/array of zeros and ones, or cplex SolveSolution object.")


    def check_solution(self, verbose: bool = False) -> ErrorDict:
        error: dict[str, list[str]] = {"C1": [], "C3": [], "C4": [], "C5": []}
        error = self.check_c1(error, verbose)
        error = self.check_c3(error, verbose)
        error = self.check_c4(error, verbose)
        error = self.check_c5(error, verbose)
        return error


    def get_objective(self) -> float:
        objective = 0
        for machine, job_list in enumerate(self.solution.values()):
            machine_busy_time = 0
            # Sort the jobs by their start time
            job_list = sorted(job_list, key=lambda x: x[1])
            for job, timestep in job_list:
                if timestep > 0:
                    try:
                        previous_job = job_list[timestep - 1][0]
                        objective -= self.data.alpha * self.data.setup_times[previous_job][job]
                    except IndexError:
                        # An index error hints that a timestep has been skipped which means no setup time added to the objective.
                        # This is a constraint violation which we are not penalizing.
                        pass
                objective += self.data.job_values[job][machine]
                machine_busy_time += self.data.processing_times[job]
            objective -= self.data.beta * (machine_busy_time ** 2)
        return -objective



    def decode_solution_dict(self, solution: dict[str, int]) -> SolutionType:
        """
        Decodes solutions given in a dictionary where the keys are the variables given as x_j_m_n and the values are if the variable
            is a part of the solution or not.
        """
        solutions: list[list[tuple[int, int]]] = [[] for _ in range(self.data.m)]
        for var, value in solution.items():
            if value == 1:
                if isinstance(var, str):
                    numbers = var.split("_")[1:]
                    j, m, n = map(int, numbers)
                else:
                    j, m, n = var
                solutions[m].append((j, n))
        solutions_sorted = [sorted(sol, key=lambda x: x[1]) for sol in solutions]
        solution_dict = {f"machine_{m}": tuple(solutions_sorted[m]) for m in range(self.data.m)}
        return solution_dict

    def decode_solution_array(self, solution: Union[list[int], npt.NDArray[int]]) -> SolutionType:
        """
        Decodes solutions given in a list where the index of the list is the job and the value is the machine.
        """
        solutions: list[list[tuple[int, int]]] = [[] for _ in range(self.data.m)]
        for q, variable in enumerate(solution):
            if variable == 1:
                j, m, n = self.data.q_to_jmn(q)
                solutions[m].append((j, n))
        solutions_sorted = [sorted(sol, key=lambda x: x[1]) for sol in solutions]
        solution_dict = {f"machine_{m}": tuple(solutions_sorted[m]) for m in range(self.data.m)}
        return solution_dict
    
    def to_cplex_format(self):
        solution = {}
        for machine, schedule in self.solution.items():
            m = int(machine[8])
            for s in schedule:
                j, n = s[0], s[1]
                solution[f"x_{j}_{m}_{n}"] = 1
        return solution



    def check_c1(self, error: dict[str, list[str]], verbose: bool = False):
        for machine, job_list in self.solution.items():
            # Extract the machine number as an int from the string "machine_i"
            machine = int(findall(r'\d+', machine)[0])
            for job, timestep in job_list:
                if machine not in self.data.eligible_machines[job]:
                    if verbose:
                        print(f"False C1: job: {job} not doable in machine {machine}")
                    error["C1"].append(f". False C1: job: {job} not doable in machine {machine}" + "<br>")
        return error

    def check_c3(self, error: dict[str, list[str]], verbose: bool = False):
        for machine, job_list in self.solution.items():
            timesteps = [timestep for job, timestep in job_list]
            timesteps_set = set(timesteps)
            if len(timesteps) != len(timesteps_set):
                if verbose:
                    print(f"False C3: {timesteps} in machine {machine}")
                error["C3"].append(f"False C3: repeated timestep in machine {machine}: {timesteps}" + "<br>")
        return error

    def check_c4(self, error: dict[str, list[str]], verbose: bool = False):
        for machine, job_list in self.solution.items():
            timesteps = [timestep for job, timestep in job_list]
            if len(timesteps) != 0:
                for step in range(max(timesteps)):
                    if step not in timesteps:
                        if verbose:
                            print(f"False C4: timestep: {step} not done by {machine}")
                        error["C4"].append(f"False C4: timestep: {step} not done by {machine}" + "<br>")
        return error

    def check_c5(self, error: dict[str, list[str]], verbose: bool = False):
        for job in range(self.data.j):
            found_job = False
            for machine, job_list in self.solution.items():
                for job_i, timestep in job_list:
                    if job == job_i:
                        if found_job:
                            if verbose:
                                print(f"False C5: job: {job} done twice")
                            error["C5"].append(f"False C5: job: {job} done twice" + "<br>")
                        else:
                            found_job = True
            if not found_job:
                if verbose:
                    print(f"False C5: job: {job} not done")
                error["C5"].append(f"False C5: job: {job} not done" + "<br>")
        return error

