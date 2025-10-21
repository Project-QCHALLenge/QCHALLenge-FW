import math
import unittest
import networkx as nx
from unittest import TestCase, mock
import unittest.mock
import random
from sp.models.sp_qubo_binary import QuboSPBinary
from sp.models.sp_qaoansatz import SPQAOAnsatz
import itertools
import numpy as np
from sp.evaluation.evaluation import SPEvaluation
from dwave.samplers import SimulatedAnnealingSampler


class MyTestCase(unittest.TestCase):

    @mock.patch("sp.evaluation.evaluation.SPEvaluation._SPEvaluation__generateOptimizedGraph")
    @mock.patch("sp.evaluation.evaluation.SPEvaluation.create_optimized_connections")
    @mock.patch("sp.models.sp_qubo_binary.QuboSPBinary._QuboSPBinary__weight_calculation")
    def test_star_graph(self, mock_weight, mock_create_optimized_connections, mock_generateOPtimizedGraph):
        mock_weight.return_value = 1
        mock_generateOPtimizedGraph.return_value = None
        mock_create_optimized_connections.return_value = None
        number_of_vertices = random.randint(4, 10)
        star_graph = nx.star_graph(number_of_vertices)
        star_graph.add_node(number_of_vertices + 1)
        star_graph.add_edges_from(itertools.product([number_of_vertices + 1], range(1, number_of_vertices+1)))
        mock_data = mock.Mock()
        map = {x: (0, 0, x) for x in range(1, number_of_vertices + 1)}
        map[0] = (0,0,0,0,0)
        map[number_of_vertices + 1] = (0, 0, 0, 0, 1)
        star_graph = nx.relabel_nodes(star_graph, map)
        mock_data.G = star_graph
        mock_data.problem_dict = mock_data.__dict__
        mock_data.listLidar = [(0, 0, 0, 0, 0), (0, 0, 0, 0, 1)] # Central vertex
        mock_data.listStreetPoints = [(0,0,x) for x in range(1, number_of_vertices+1)]
        mock_data.walls = []
        model = SPQAOAnsatz(mock_data)
        solution = model.solve(iterations=10, optimizer="Adam", learning_rate=0.1, info=True)["solution"]
        objective_value = SPEvaluation(mock_data, solution).get_objective()
        gap = abs(1 - objective_value) / abs(objective_value)
        self.assertLessEqual(gap, 0.5)  # add assertion here
    @mock.patch("sp.evaluation.evaluation.SPEvaluation._SPEvaluation__generateOptimizedGraph")
    @mock.patch("sp.evaluation.evaluation.SPEvaluation.create_optimized_connections")
    @mock.patch("sp.models.sp_qubo_binary.QuboSPBinary._QuboSPBinary__weight_calculation")
    def test_path_graph(self, mock_weight, mock_create_optimized_connections, mock_generateOPtimizedGraph):
        mock_weight.return_value = 1
        mock_generateOPtimizedGraph.return_value = None
        mock_create_optimized_connections.return_value = None
        number_of_vertices = 2 * random.randint(2, 5) + 1
        print(number_of_vertices)
        path = nx.path_graph(number_of_vertices)
        lidar_indices = math.floor(number_of_vertices / 2)
        optimal_number_of_selected_indices = math.ceil(lidar_indices / 2)
        mock_data = mock.Mock()
        map = {x: (0, 0, 0, 0, x) for x in range(number_of_vertices + 1) if x % 2 == 0}
        map.update({x: (0, 0, x) for x in range(number_of_vertices + 1) if x % 2 == 1})
        path = nx.relabel_nodes(path, map)
        mock_data.G = path
        mock_data.walls = []
        mock_data.listLidar = [(0, 0, 0, 0, x) for x in range(number_of_vertices) if x % 2 == 0]
        mock_data.listStreetPoints = [(0,0,x) for x in range(number_of_vertices) if x % 2 == 1]
        model = SPQAOAnsatz(mock_data)
        solution = model.solve(iterations=170, optimizer="Adam", learning_rate=0.01, seed=501, info=True)["solution"]
        objective_value = SPEvaluation(mock_data, solution).get_objective()
        gap = abs(optimal_number_of_selected_indices - objective_value) / abs(objective_value)
        self.assertLessEqual(gap, 0.5)  # add assertion here


    @mock.patch("sp.evaluation.evaluation.SPEvaluation._SPEvaluation__generateOptimizedGraph")
    @mock.patch("sp.evaluation.evaluation.SPEvaluation.create_optimized_connections")
    @mock.patch("sp.models.sp_qubo_binary.QuboSPBinary._QuboSPBinary__weight_calculation")
    def test_m_m_bipartite_graph(self,mock_weight, mock_create_optimized_connections, mock_generateOPtimizedGraph):
        mock_weight.return_value = 1
        mock_generateOPtimizedGraph.return_value = None
        mock_create_optimized_connections.return_value = None
        number_of_vertices = random.randint(4, 6)
        a_degree = random.randint(2, math.floor(number_of_vertices/2))
        edges = []
        b_vertex_indices = [x for x in range(number_of_vertices)]
        b_degrees = {b_vertex: 0 for b_vertex in b_vertex_indices}
        a_vertices = [(0, 0, 0, 0, x) for x in range(number_of_vertices)]
        b_vertices = [(0, 0, x) for x in range(number_of_vertices)]
        for a_vertex in a_vertices:
            selected_b_vertex_indices = np.random.choice(b_vertex_indices, size=a_degree, replace=False)
            for index in selected_b_vertex_indices:
                b_degrees[index] += 1
            b_vertex_selection = [b_vertices[x] for x in selected_b_vertex_indices]
            edges += list(itertools.product([a_vertex], b_vertex_selection))
        for b_vertex in b_vertex_indices:
            if b_degrees[b_vertex] <= 1:
                a_vertex_1 = a_vertices[-1]
                a_vertex_2 = a_vertices[0]
                edges.append((a_vertex_1, b_vertices[b_vertex]))
                edges.append((a_vertex_2, b_vertices[b_vertex]))
                a_degree += 1

        bipartite_graph = nx.Graph()
        bipartite_graph.add_nodes_from(a_vertices, bipartite=0)
        bipartite_graph.add_nodes_from(b_vertices, bipartite=1)
        bipartite_graph.add_edges_from(edges)
        mock_data = mock.Mock()
        mock_data.walls = []
        mock_data.G = bipartite_graph
        mock_data.listLidar = a_vertices
        mock_data.listStreetPoints = b_vertices
        model = SPQAOAnsatz(mock_data)
        solution = model.solve(iterations=40, optimizer="Adam", learning_rate=0.01, seed=501, info=True)["solution"]
        objective_value = SPEvaluation(mock_data, solution).get_objective()
        lower_bound = np.floor(number_of_vertices / a_degree)
        gap = abs(lower_bound - objective_value) / abs(objective_value)
        self.assertLessEqual(gap, 1)  # add assertion heree

if __name__ == '__main__':


    unittest.main()
