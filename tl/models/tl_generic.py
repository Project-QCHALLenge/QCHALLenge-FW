# local imports
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from utils.tl_generic_model import TruckLoadingModel

import itertools

class Tl2D_Generic(TruckLoadingModel):

    data = None
    model_name: str = "Generic"
    boxes = list

    def __init__(self, data):
        self.data = data
        self.boxes = data.get_box_names_as_list()
        self.permuted_box = list(itertools.permutations(self.boxes, 2))

    def optimize(self):
        """
        The optimize functions needs to be declared in submodule
        :return:
        """
        return NotImplementedError
