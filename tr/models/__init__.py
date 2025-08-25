import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, parent_dir)

from tr.models.tr_cplex import TR_cplex as TRCplex
from tr.models.tr_gurobi import GurobiTR as TRGurobi


__all__ = ["TRCplex", "TRGurobi"]
