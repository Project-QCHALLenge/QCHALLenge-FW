import unittest

from mpl import MPLGurobiNoWaitOverlap, MPLGurobiNoWaitOverlapReduced, MPLCQMNoWaitOverlap
from mpl.data.mpl_data import MPLData
from mpl.evaluation.evaluation import MPLEvaluation
from mpl.plotting.mpl_plot import MPLPlot
from gurobipy import GRB
import gurobipy as gp


class TestNoWaitOverlap(unittest.TestCase):
    def test_GurobiReduced_infeasible(self):
        params = {"N_A": 1, "N_B": 1, "R": 1, "t_r" : 1, "T": 40, "processing_times": {'Jobs_A': (2, 2), 'Jobs_B': (2, 2, 2)}}
        problem = MPLData.from_dict(params)
        model = MPLGurobiNoWaitOverlapReduced(problem)
        model.model.addConstr(model.x[ 1, 1, 0, 0] == 1)
        model.model.setParam("TimeLimit", 300)
        model.solve()
        self.assertNotEqual(model.model.status, GRB.TIME_LIMIT)
        self.assertEqual(model.model.status,  GRB.INFEASIBLE)

    def test_GurobiReduced_objective(self):
        params = {"N_A": 1, "N_B": 1, "R": 1, "t_r": 1, "T": 40,
                  "processing_times": {'Jobs_A': (2, 2), 'Jobs_B': (2, 2, 2)}}
        problem = MPLData.from_dict(params)
        model = MPLGurobiNoWaitOverlapReduced(problem)
        model.model.setParam("TimeLimit", 300)
        answer = model.solve()
        evaluation = MPLEvaluation(problem, answer["solution"])
        objective = evaluation.get_objective()
        self.assertNotEqual(model.model.status, GRB.TIME_LIMIT)
        self.assertEqual(16, objective)

    def test_Gurobi_infeasible(self):
        params = {"N_A": 1, "N_B": 1, "R": 1, "t_r" : 1, "T": 40, "processing_times": {'Jobs_A': (2, 2), 'Jobs_B': (2, 2, 2)}}
        problem = MPLData.from_dict(params)
        model = MPLGurobiNoWaitOverlap(problem)
        model.model.setParam("TimeLimit", 300)
        model.model.addConstr(model.x[1, 0, 0, 1, 1] == 1)
        model.model.optimize()
        self.assertNotEqual(model.model.status, GRB.TIME_LIMIT)
        self.assertEqual(model.model.status,  GRB.INFEASIBLE)

    def test_Gurobi_objective(self):
        params = {"N_A": 1, "N_B": 1, "R": 1, "t_r" : 1, "T": 40, "processing_times": {'Jobs_A': (2, 2), 'Jobs_B': (2, 2, 2)}}
        problem = MPLData.from_dict(params)
        model = MPLGurobiNoWaitOverlap(problem)
        model.model.setParam("TimeLimit", 300)
        answer = model.solve()
        evaluation = MPLEvaluation(problem, answer["solution"])
        objective = evaluation.get_objective()
        self.assertNotEqual(model.model.status, GRB.TIME_LIMIT)
        self.assertEqual(16, objective)


    def test_CQM_infeasible(self):
        params = {"N_A": 1, "N_B": 1, "R": 1, "t_r" : 1, "T": 40, "processing_times": {'Jobs_A': (2, 2), 'Jobs_B': (2, 2, 2)}}
        problem = MPLData.from_dict(params)
        cqm_model = MPLCQMNoWaitOverlap(problem)
        model = gp.read(cqm_model.tmp_output_path.as_posix())
        model.setParam("TimeLimit", 300)
        model.addConstr(model.getVarByName("x_1_0_0_1_1") == 1)
        model.optimize()
        self.assertNotEqual(model.status, GRB.TIME_LIMIT)
        self.assertEqual(model.status,  GRB.INFEASIBLE)

    def test_CQM_objective(self):
        params = {"N_A": 1, "N_B": 1, "R": 1, "t_r" : 1, "T": 40, "processing_times": {'Jobs_A': (2, 2), 'Jobs_B': (2, 2, 2)}}
        problem = MPLData.from_dict(params)
        cqm_model = MPLCQMNoWaitOverlap(problem)
        model = gp.read(cqm_model.tmp_output_path.as_posix())
        model.setParam("TimeLimit", 300)
        model.optimize()
        self.assertNotEqual(model.status, GRB.TIME_LIMIT)
        self.assertEqual(16, model.ObjVal)


if __name__ == '__main__':
    unittest.main()
