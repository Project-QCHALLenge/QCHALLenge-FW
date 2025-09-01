from abc import ABC, abstractmethod


class AbstractModel:
    @abstractmethod
    def solve(self, *args, **kwargs):
        return

    @abstractmethod
    def build_model(self, *args, **kwargs):
        return

