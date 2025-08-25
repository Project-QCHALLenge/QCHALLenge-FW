# can be deleted if we use packages
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

from lb.models.lb_gurobi_cqm_unary_uniform_truck_capacity import LBGurobiCQMUnaryUniformTruckCapacity
from lb.models.lb_gurobi_qubo_unary_uniform_truck_capacity import LBGurobiQUBOUnaryUniformTruckCapacity
from lb.models.lb_dwave_cqm_unary_uniform_truck_capacity import LBDWAVECQMUnaryUniformTruckCapacity
from lb.models.lb_dwave_qubo_unary_uniform_truck_capacity import LBDWAVEQUBOUnaryUniformTruckCapacity