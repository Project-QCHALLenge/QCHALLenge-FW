import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils.tl_interpretation_generic import Tl_Generic_Solution


class Tl2D_Solution(Tl_Generic_Solution):
    """
    This class contains all the __init__ and run information for a specific optimization.
    """

    def to_dict(self):

        return {
            "model_name": self.model_name,
            "selected_boxes": self.selected_boxes,
            "runtime": self.runtime,
            "num_vars": self.num_vars,
            "num_constrs": self.num_constrs,
            "objective_value": self.objective_value,
            "gap_to_optimal": self.gap_to_optimal,
            "violated_constraints": self.violated_constraints,
            "energy": self.energy,
        }
