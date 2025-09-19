import unittest

from mpl import MPLGurobiWaitOverlap
from mpl.data.mpl_data import MPLData
from mpl.evaluation.evaluation import MPLEvaluation
from gurobipy import GRB


class TestNoWaitOverlap(unittest.TestCase):

    def test_GurobiReduced_infeasible(self):
        params = {"N_A": 1, "N_B": 1, "R": 1, "t_r": 1, "T": 40,
                  "processing_times": {'Jobs_A': (2, 2), 'Jobs_B': (2, 2, 2)}}
        problem = MPLData.from_dict(params)
        model = MPLGurobiWaitOverlap(problem)
        model.model.addConstr(model.x[1, 0, 0, 1, 1] == 1)
        model.model.setParam("TimeLimit", 300)
        model.solve()
        self.assertNotEqual(model.model.status, GRB.TIME_LIMIT)
        self.assertNotEquals(model.model.status, GRB.INFEASIBLE)

    def test_GurobiReduced_objective(self):
        params = {"N_A": 1, "N_B": 1, "R": 1, "t_r": 1, "T": 40,
                  "processing_times": {'Jobs_A': (2, 2), 'Jobs_B': (2, 2, 2)}}
        problem = MPLData.from_dict(params)
        model = MPLGurobiWaitOverlap(problem)
        model.model.setParam("TimeLimit", 300)
        answer = model.solve()
        evaluation = MPLEvaluation(problem, answer["solution"])
        objective = evaluation.get_objective()
        self.assertNotEqual(model.model.status, GRB.TIME_LIMIT)
        self.assertEqual(16, objective)




if __name__ == '__main__':
    unittest.main()
