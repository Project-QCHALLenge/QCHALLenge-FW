from abc import ABC, abstractmethod


class AbstractData(ABC):

    @abstractmethod
    def create_problem(cls,  *args, **kwargs):
        pass

    @abstractmethod
    def from_json(cls,  *args, **kwargs):
        pass

    @abstractmethod
    def from_random(cls, *args, **kwargs):
        pass