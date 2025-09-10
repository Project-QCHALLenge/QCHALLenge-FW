import math
import unittest
import networkx as nx
from unittest import TestCase, mock
import unittest.mock
import random
from sp.models.sp_qubo_onehot import QuboSPOnehot
import itertools
import numpy as np
from sp.evaluation.evaluation import SPEvaluation
from dwave.samplers import SimulatedAnnealingSampler
import dimod
import gurobipy as gp
from gurobipy import GRB


class TestQuboOneHot(unittest.TestCase):
    @staticmethod
    def create_gurobi_model_form_Q(Q):
        model = gp.Model("QUBO")
        variables = model.addVars(range(Q.shape[0]), vtype=GRB.BINARY)
        x = gp.MVar.fromlist(list(variables.values()))
        model.setObjective(x.T @ Q @ x, GRB.MINIMIZE)
        return model

    @staticmethod
    def solve_function(Q, *args, **kwargs):
        model = TestQuboOneHot.create_gurobi_model_form_Q(Q)
        model.optimize()
        all_vars = model.getVars()
        values = model.getAttr("X", all_vars)
        names = model.getAttr("VarName", all_vars)
        answer = np.zeros(shape=(Q.shape[0]))
        for name, val in zip(names, values):
            index = int(name.replace("C", ""))
            answer[index] = val
        answer_object = dimod.SampleSet.from_samples(answer, "BINARY", model.ObjVal)
        return answer_object
    @mock.patch("sp.evaluation.evaluation.SPEvaluation._SPEvaluation__generateOptimizedGraph")
    @mock.patch("sp.evaluation.evaluation.SPEvaluation.create_optimized_connections")
    def test_star_graph(self, mock_create_optimized_connections, mock_generateOPtimizedGraph):
        mock_generateOPtimizedGraph.return_value = None
        mock_create_optimized_connections.return_value = None
        number_of_vertices = random.randint(3, 25)
        star_graph = nx.star_graph(number_of_vertices)
        mock_data = mock.Mock()
        map = {x: (0, 0, x) for x in range(1, number_of_vertices + 1)}
        map[0] = (0,0,0,0,0)
        star_graph = nx.relabel_nodes(star_graph, map)
        mock_data.G = star_graph
        mock_data.problem_dict = mock_data.__dict__
        mock_data.listLidar = [(0, 0, 0, 0, 0)] # Central verted
        mock_data.listStreetPoints = [(0,0,x) for x in range(1, number_of_vertices+1)]
        model = QuboSPOnehot(mock_data)
        solution = model.solve(TestQuboOneHot.solve_function)["solution"]
        objective_value = SPEvaluation(mock_data, solution).get_objective()
        self.assertEqual(objective_value, 1)  # add assertion here

    @mock.patch("sp.evaluation.evaluation.SPEvaluation._SPEvaluation__generateOptimizedGraph")
    @mock.patch("sp.evaluation.evaluation.SPEvaluation.create_optimized_connections")
    def test_path_graph(self, mock_create_optimized_connections, mock_generateOPtimizedGraph):
        mock_generateOPtimizedGraph.return_value = None
        mock_create_optimized_connections.return_value = None
        number_of_vertices = random.randint(3, 25)
        path = nx.path_graph(number_of_vertices)
        lidar_indices = math.floor(number_of_vertices / 2)
        optimal_number_of_selected_indices = math.ceil(lidar_indices / 2)
        mock_data = mock.Mock()
        map = {x: (0, 0, 0, 0, x) for x in range(number_of_vertices + 1) if x % 2 == 0}
        map.update({x: (0, 0, x) for x in range(number_of_vertices + 1) if x % 2 == 1})
        path = nx.relabel_nodes(path, map)
        mock_data.G = path
        mock_data.listLidar = [(0, 0, 0, 0, x) for x in range(number_of_vertices) if x % 2 == 0]
        mock_data.listStreetPoints = [(0,0,x) for x in range(number_of_vertices) if x % 2 == 1]
        model = QuboSPOnehot(mock_data)
        solution = model.solve(TestQuboOneHot.solve_function)["solution"]
        objective_value = SPEvaluation(mock_data, solution).get_objective()
        self.assertEqual(optimal_number_of_selected_indices, objective_value)  # add assertion here


    @mock.patch("sp.evaluation.evaluation.SPEvaluation._SPEvaluation__generateOptimizedGraph")
    @mock.patch("sp.evaluation.evaluation.SPEvaluation.create_optimized_connections")
    def test_pathological_path_graph(self, mock_create_optimized_connections, mock_generateOPtimizedGraph):
        mock_generateOPtimizedGraph.return_value = None
        mock_create_optimized_connections.return_value = None
        number_of_A_vertices =  2*random.randint(1, 25)
        number_of_B_vertices =  number_of_A_vertices + 1
        number_of_vertices = number_of_B_vertices + number_of_A_vertices
        path = nx.path_graph(number_of_vertices)
        lidar_indices = math.floor(number_of_vertices / 2)
        optimal_number_of_selected_indices = math.ceil(lidar_indices / 2) + 1
        mock_data = mock.Mock()
        map = {x: (0, 0, 0, 0, x) for x in range(number_of_vertices + 1) if x % 2 == 1}
        map.update({x: (0, 0, x) for x in range(number_of_vertices + 1) if x % 2 == 0})
        path = nx.relabel_nodes(path, map)
        mock_data.G = path
        mock_data.listLidar = [(0, 0, 0, 0, x) for x in range(number_of_vertices) if x % 2 == 1]
        mock_data.listStreetPoints = [(0,0,x) for x in range(number_of_vertices) if x % 2 == 0]
        model = QuboSPOnehot(mock_data)
        solution = model.solve(TestQuboOneHot.solve_function)["solution"]
        objective_value = SPEvaluation(mock_data, solution).get_objective()
        self.assertEqual(optimal_number_of_selected_indices, objective_value)  # add assertion here

    @mock.patch("sp.evaluation.evaluation.SPEvaluation._SPEvaluation__generateOptimizedGraph")
    @mock.patch("sp.evaluation.evaluation.SPEvaluation.create_optimized_connections")
    def test_m_m_bipartite_graph(self, mock_create_optimized_connections, mock_generateOPtimizedGraph):
        mock_generateOPtimizedGraph.return_value = None
        mock_create_optimized_connections.return_value = None
        number_of_vertices = random.randint(3, 25)
        a_degree = random.randint(1, math.floor(number_of_vertices/2))
        edges = []
        b_vertex_indices = [x for x in range(number_of_vertices)]
        already_selected_b_indices = []
        a_vertices = [(0, 0, 0, 0, x) for x in range(number_of_vertices)]
        b_vertices = [(0, 0, x) for x in range(number_of_vertices)]
        for a_vertex in a_vertices:
            selected_b_vertex_indices = np.random.choice(b_vertex_indices, size=a_degree, replace=False)
            already_selected_b_indices += list(selected_b_vertex_indices)
            b_vertex_selection = [b_vertices[x] for x in selected_b_vertex_indices]
            edges += list(itertools.product([a_vertex], b_vertex_selection))
        b_vertices_without_edges = list(set(b_vertex_indices) - set(already_selected_b_indices))
        for b_vertex in b_vertices_without_edges:
            a_vertex = a_vertices[b_vertex]
            edges.append((a_vertex, b_vertices[b_vertex]))
            a_degree += 1

        bipartite_graph = nx.Graph()
        bipartite_graph.add_nodes_from(a_vertices, bipartite=0)
        bipartite_graph.add_nodes_from(b_vertices, bipartite=1)
        bipartite_graph.add_edges_from(edges)
        mock_data = mock.Mock()
        mock_data.G = bipartite_graph
        mock_data.listLidar = a_vertices
        mock_data.listStreetPoints = b_vertices
        model = QuboSPOnehot(mock_data)
        solution = model.solve(TestQuboOneHot.solve_function)["solution"]
        objective_value = SPEvaluation(mock_data, solution).get_objective()
        lower_bound = np.floor(number_of_vertices / a_degree)
        self.assertLessEqual(lower_bound, objective_value)  # add assertion here

if __name__ == '__main__':
    unittest.main()
