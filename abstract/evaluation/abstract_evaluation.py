from abc import ABC, abstractmethod


class AbstractEvaluation:

    @abstractmethod
    def get_objective(self):
        return

    @abstractmethod
    def check_solution(self):
        return
