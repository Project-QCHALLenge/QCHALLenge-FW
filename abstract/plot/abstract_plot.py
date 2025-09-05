from abc import ABC, abstractmethod


class AbstractPlot(ABC):
    @abstractmethod
    def plot_solution(self, *args, **kwargs):
        return
