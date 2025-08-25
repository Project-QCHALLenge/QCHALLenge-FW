from abc import ABC, abstractmethod
from lb.data.lb_data import LBData


class LBAbstractModel(ABC):
    def __init__(self, data: LBData):
        self.data = data
        self.Q, self.A, self.rhs = None, None, None
        self.decision_variables = None
        self.model = None
        pass

    @abstractmethod
    def solve(self, **params):
        pass

    @abstractmethod
    def solution(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def _create_variables(self):
        pass

    @abstractmethod
    def _create_data_structures(self):
        pass

    @abstractmethod
    def set_initial_solution(self, solution):
        pass
