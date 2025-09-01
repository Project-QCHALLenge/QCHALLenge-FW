from tr.data.tr_data import TRData
from tr.models.tr_cplex import TR_cplex

from typing import Union
from abstract.evaluation.abstract_evaluation import AbstractEvaluation


class TREvaluation(AbstractEvaluation):

    def __init__(
        self,
        data: TRData,
        solution: dict[str, int],
    ):
        self.data = data
        self.cpmodel = TR_cplex(data)
        self.solution = solution

    def set_solution(self, solution: dict[str, int]):

        self.solution = solution

    def check_solution(self, verbose: bool = False):
        return self.cpmodel.check_violation(self.solution)

    def get_objective(self) -> float:
        objective = self.cpmodel.objectiv_from_solution(
            self.solution, only_feasible=False
        )
        return objective
