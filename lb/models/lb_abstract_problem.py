from abc import ABC, abstractmethod
from lb.data.lb_data import LBData


class LBAbstractProblem(ABC):
    def __init__(self, data: LBData):
        self.data = data
        self.Q = None
        self.c = None
        self.A = None
        self.rhs = None
        self._create_data_structures()

    def assemble(self):
        self._assemble_constraints()
        self._assemble_objective_function()
        return self.Q, self.A, self.rhs

    @abstractmethod
    def _assemble_objective_function(self):
        pass

    @abstractmethod
    def _assemble_constraints(self):
        pass

    @abstractmethod
    def _create_data_structures(self):
        pass
