from abc import ABC, abstractmethod


class AbstractData(ABC):

    @abstractmethod
    def create_problem(self,  *args, **kwargs):
        return


