import unittest

from tr.data.railnetwork import *
from tr.data.tr_data import TRData
from tr.models.tr_cplex import TR_cplex
import networkx as nx
from tr.evaluation.evaluation import TREvaluation
from gurobipy import GRB
import numpy as np
from docplex.util.status import JobSolveStatus
from tr.plotting.tr_plot import TRPlot


class MyTestCase(unittest.TestCase):
    def test_trivial(self):
        stations = ["A", "B", "C"]
        rail_graph = nx.Graph()
        rail_graph.add_nodes_from(stations)
        rail_graph.add_edge("A", "C", distance=10, max_speed=10)
        rail_graph.add_edge("B", "C", distance=10, max_speed=10)
        train_1 = Train((("A", 0, 1), ("C", 2, 3)), speed=10)
        train_2 = Train((("B", 0, 1), ("C", 2, 3)), speed=10)


        rail_network = RailNetwork(rail_graph, [train_1, train_2])
        data = TRData(rail_network)

        model = TR_cplex(data)
        model._model.set_time_limit(300)
        answer = model.solve()["solution"]
        evaluation = TREvaluation(data, answer)

        self.assertNotEqual(model._model.solve_details.status_code, 107)
        self.assertEqual(evaluation.get_objective(), 0)  # add assertion here

    def test_opposing_stations(self):
        stations = ["A", "B"]
        rail_graph = nx.Graph()
        rail_graph.add_nodes_from(stations)
        rail_graph.add_edge("A", "B", distance=10, max_speed=10)
        train_1 = Train((("A", 0, 1), ("B", 2, 3)), speed=10)
        train_2 = Train((("B", 0, 1), ("A", 2, 3)), speed=10)


        rail_network = RailNetwork(rail_graph, [train_1, train_2])
        data = TRData(rail_network)

        model = TR_cplex(data)
        model._model.set_time_limit(300)
        answer = model.solve()["solution"]
        evaluation = TREvaluation(data, answer)

        self.assertNotEqual(model._model.solve_details.status_code, 107)
        self.assertEqual(evaluation.get_objective(), 2)  # add assertion here

    def test_circular_schedule(self):
        stations = ["A", "B", "C"]
        rail_graph = nx.Graph()
        rail_graph.add_nodes_from(stations)
        rail_graph.add_edge("A", "B", distance=10, max_speed=10)
        rail_graph.add_edge("B", "C", distance=10, max_speed=10)
        rail_graph.add_edge("C", "A", distance=10, max_speed=10)
        train_1 = Train((("A", 0, 1), ("B", 2, 3)), speed=10)
        train_2 = Train((("B", 0, 1), ("C", 2, 3)), speed=10)
        train_3 = Train((("C", 0, 1), ("A", 2, 3)), speed=10)

        rail_network = RailNetwork(rail_graph, [train_1, train_2, train_3])
        data = TRData(rail_network)

        model = TR_cplex(data)
        model._model.set_time_limit(300)
        answer = model.solve()["solution"]
        evaluation = TREvaluation(data, answer)

        self.assertNotEqual(model._model.solve_details.status_code, 107)
        self.assertEqual(evaluation.get_objective(), 0)  # add assertion here

    def test_indirect_route(self):
        stations = ["A", "B", "C"]
        rail_graph = nx.Graph()
        rail_graph.add_nodes_from(stations)
        rail_graph.add_edge("A", "B", distance=10, max_speed=10)
        rail_graph.add_edge("B", "C", distance=10, max_speed=10)
        train_1 = Train((("A", 0, 1), ("B", 2, 2), ("C", 3, 4)), speed=10)

        rail_network = RailNetwork(rail_graph, [train_1])
        data = TRData(rail_network)

        model = TR_cplex(data)
        model._model.set_time_limit(300)
        answer = model.solve()["solution"]
        evaluation = TREvaluation(data, answer)

        self.assertNotEqual(model._model.solve_details.status_code, 107)
        self.assertEqual(evaluation.get_objective(), 0)  # add assertion here

    def test_invalid_schedule(self):
        stations = ["A", "B", "C"]
        rail_graph = nx.Graph()
        rail_graph.add_nodes_from(stations)
        rail_graph.add_edge("A", "B", distance=10, max_speed=10)
        rail_graph.add_edge("B", "C", distance=10, max_speed=10)
        train_1 = Train((("A", 0, 1), ("B", 0, -1), ("C", -1, -2)), speed=10)

        rail_network = RailNetwork(rail_graph, [train_1])
        data = TRData(rail_network)

        model = TR_cplex(data)
        model._model.set_time_limit(300)
        model._model.solve()

        self.assertNotEqual(model._model.solve_details.status_code, 107)
        self.assertEqual(model._model.solve_status, JobSolveStatus.INFEASIBLE_SOLUTION)  # add assertion here

    def test_integer_stations(self):
        stations = [0, 1, 2]
        rail_graph = nx.Graph()
        rail_graph.add_nodes_from(stations)
        rail_graph.add_edge(0, 1, distance=10, max_speed=10)
        rail_graph.add_edge(1, 2, distance=10, max_speed=10)
        train_1 = Train(stations, speed=10)

        rail_network = RailNetwork(rail_graph, [train_1])
        data = TRData(rail_network)

        model = TR_cplex(data)
        model._model.set_time_limit(300)
        answer = model.solve()["solution"]
        evaluation = TREvaluation(data, answer)
        try:
            TRPlot(evaluation)
        except KeyError:
            self.fail("KeyError")

    def test_headway_fail(self):
        stations = ["A", "B"]
        rail_graph = nx.Graph()
        rail_graph.add_nodes_from(stations)
        rail_graph.add_edge("A", "B", distance=7, max_speed=90)
        train_1 = Train(["A", "B"], speed=1)
        train_2 = Train(["A", "B"], speed=1)

        rail_network = RailNetwork(rail_graph, [train_1, train_2])
        data = TRData(rail_network)

        model = TR_cplex(data)
        model._model.set_time_limit(300)
        model.solve()

        self.assertNotEqual(model._model.solve_details.status_code, 107)
        self.assertNotEqual(model._model.solve_status, JobSolveStatus.INFEASIBLE_SOLUTION)


    def test_opposing_stations_fail(self):
        stations = ["A", "B"]
        rail_graph = nx.Graph()
        rail_graph.add_nodes_from(stations)
        rail_graph.add_edge("A", "B", distance=7, max_speed=90)
        train_1 = Train(["A", "B"], speed=1)
        train_2 = Train(["B", "A"], speed=1)

        rail_network = RailNetwork(rail_graph, [train_1, train_2])
        data = TRData(rail_network)
        model = TR_cplex(data)
        model._model.set_time_limit(300)
        model._model.solve()

        self.assertNotEqual(model._model.solve_details.status_code, 107)
        self.assertNotEqual(model._model.solve_status, JobSolveStatus.INFEASIBLE_SOLUTION)






if __name__ == '__main__':
    unittest.main()
