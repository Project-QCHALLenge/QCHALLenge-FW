import unittest

from tr.data.railnetwork import *
from tr.data.tr_data import TRData
from tr.models.tr_gurobi import GurobiTR
import networkx as nx
from tr.evaluation.evaluation import TREvaluation
from gurobipy import GRB
import numpy as np

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

        model = GurobiTR(data)
        answer = model.solve()["solution"]
        evaluation = TREvaluation(data, answer)


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

        model = GurobiTR(data)
        answer = model.solve()["solution"]
        evaluation = TREvaluation(data, answer)

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

        model = GurobiTR(data)
        answer = model.solve()["solution"]
        evaluation = TREvaluation(data, answer)
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

        model = GurobiTR(data)
        answer = model.solve()["solution"]
        evaluation = TREvaluation(data, answer)
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

        model = GurobiTR(data)
        model._grb_model.optimize()
        self.assertEqual(model._grb_model.status, GRB.INFEASIBLE)  # add assertion here

    def test_integer_stations(self):
        stations = [0, 1, 2]
        rail_graph = nx.Graph()
        rail_graph.add_nodes_from(stations)
        rail_graph.add_edge(0, 1, distance=10, max_speed=10)
        rail_graph.add_edge(1, 2, distance=10, max_speed=10)
        train_1 = Train(stations, speed=10)

        rail_network = RailNetwork(rail_graph, [train_1])
        data = TRData(rail_network)

        model = GurobiTR(data)
        answer = model.solve()["solution"]
        evaluation = TREvaluation(data, answer)
        try:
            TRPlot(evaluation)
        except KeyError:
            self.fail("KeyError")

    def test_random(self):
        for i in range(100):
            n_stations = 2
            n_trains = 2
            stations = range(n_stations)
            n_edges = 1

            print(f"n_edges: {n_edges}, n_stations: {n_stations}, n_trains: {n_trains}")

            rail_graph = nx.Graph()
            stations = [str(s) for s in stations]
            rail_graph.add_nodes_from(stations)
            all_edges = np.array([np.random.choice(stations, (1, 2), replace=False) for _ in range(n_edges)]).reshape(-1, 2)
            for u, v in all_edges:
                max_speed = 4 #np.random.randint(1, 10)
                distance = 25 #np.random.randint(0.5*(2*n_trains + 3) * max_speed, (2*n_trains + 3) * max_speed)
                print(f"Distance {distance}, max_speed: {max_speed}")
                rail_graph.add_edge(u, v, distance=distance, max_speed=max_speed)

            trains = []

            for train in range(n_trains):
                start, end = np.random.choice(stations, size=2, replace=False)
                routes = list(nx.all_simple_paths(rail_graph, start, end))

                if len(routes) == 0:
                    continue
                selected_route_index = np.random.choice(range(len(routes)))
                print(f"route on train {train}: {routes[selected_route_index]}, speed: {100}")
                trains.append(Train(routes[selected_route_index], speed=10))


            rail_network = RailNetwork(rail_graph, trains)
            data = TRData(rail_network)

            model = GurobiTR(data)
            model._grb_model.optimize()
            try:
                model._grb_model.computeIIS()
                model._grb_model.write("m.ilp")
            except Exception:
                pass
            answer = model.solve()["solution"]
            #evaluation = TREvaluation(data, answer)
            #TRPlot(evaluation).plot()
            #self.assertEqual(model._grb_model.status, GRB.INFEASIBLE)  # add assertion here


    def test_fail_why(self):
        n_stations = 2
        n_trains = 2 #np.random.randint(1, 15)
        stations = range(n_stations)
        n_edges = 1 #np.random.randint(n_stations, n_stations * (n_stations - 1) /2)

        print(f"n_edges: {n_edges}, n_stations: {n_stations}, n_trains: {n_trains}")

        rail_graph = nx.Graph()
        stations = [str(s) for s in stations]
        rail_graph.add_nodes_from(stations)
        all_edges = np.array([np.random.choice(stations, (1, 2), replace=False) for _ in range(n_edges)]).reshape(-1, 2)
        for u, v in all_edges:
            max_speed = np.random.randint(1, 10)
            distance = np.random.randint(0.5*(2*n_trains + 3) * max_speed, (2*n_trains + 3) * max_speed)
            print(f"Distance {distance}, max_speed: {max_speed}")
            rail_graph.add_edge(u, v, distance=distance, max_speed=max_speed)

        trains = []

        for train in range(n_trains):
            start, end = np.random.choice(stations, size=2, replace=False)
            routes = list(nx.all_simple_paths(rail_graph, start, end))

            if len(routes) == 0:
                continue
            selected_route_index = np.random.choice(range(len(routes)))
            print(f"route on train {train}: {routes[selected_route_index]}, speed: {100}")
            trains.append(Train(routes[selected_route_index], speed=10))


        rail_network = RailNetwork(rail_graph, trains)
        data = TRData(rail_network)

        model = GurobiTR(data)
        model._grb_model.optimize()
        try:
            model._grb_model.computeIIS()
            model._grb_model.write("m.ilp")
        except Exception:
            pass
        answer = model.solve()["solution"]
        #evaluation = TREvaluation(data, answer)
        #TRPlot(evaluation).plot()
        #self.assertEqual(model._grb_model.status, GRB.INFEASIBLE)  # add assertion here

    def test_opposing_stations_fail(self):
        stations = ["A", "B"]
        rail_graph = nx.Graph()
        rail_graph.add_nodes_from(stations)
        rail_graph.add_edge("A", "B", distance=7, max_speed=90)
        train_1 = Train(["A", "B"], speed=1)
        train_2 = Train(["B", "A"], speed=1)

        rail_network = RailNetwork(rail_graph, [train_1, train_2])
        data = TRData(rail_network)

        model = GurobiTR(data)
        model._grb_model.computeIIS()
        model._grb_model.write("m.ilp")
        model._grb_model.optimize()
        self.assertNotEqual(model._grb_model.status, GRB.INFEASIBLE)





if __name__ == '__main__':
    unittest.main()
