import time

import numpy as np

from pas.data.pas_data import PASData
from pas.data.decomposition import DataDecomposer
from pas.evaluation.evaluation import EvaluationPAS


class DecompositionPAS:

    def __init__(self, data: PASData):
        self.data: PASData = data

    def solve(self, solve_func, model_func, **config):

        start_time = time.time()
        decomposer = DataDecomposer(data=self.data, max_problem_size=20)
        decomposition = decomposer.decompose()
        # print(f"number of subproblems = {len(decomposition)}")

        global_solution = {}

        # Solve the subproblems
        for subproblem_data, map_to_parent in decomposition:
            qubo_model = model_func(subproblem_data)
            qubo = qubo_model.model
            answer = solve_func(Q=qubo, **config)
            sample_list = np.array([answer.first.sample[v] for v in answer.variables])
            solution = qubo_model.decode_solution(sample_list)
        
            global_subproblem_solution = DataDecomposer.map_solution(solution, map_to_parent)
            # Append the solution of the subproblem to the global solution
            global_solution.update(global_subproblem_solution)


        # Evaluate and check the global solution
        global_evaluation = EvaluationPAS(data=self.data, solution=global_solution,convert_solution=False)
        global_solution = global_evaluation.to_cplex_format()
        runtime = time.time() - start_time

        result = {"solution": global_solution, "energy": 0, "runtime": runtime}

        return result