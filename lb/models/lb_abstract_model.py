from abc import ABC, abstractmethod
from lb.data.lb_data import LBData
from abstract.models.abstract_model import AbstractModel

class LBAbstractModel(ABC):
    def __init__(self, data: LBData):
        self.data = data
        self.Q, self.A, self.rhs = None, None, None
        self.decision_variables = None
        self.model = None
        pass

    @abstractmethod
    def solution(self, *args, **kwargs):
        pass

    @abstractmethod
    def _create_variables(self, *args, **kwargs):
        pass

    @abstractmethod
    def _create_data_structures(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_initial_solution(self, *args, **kwargs):
        pass
