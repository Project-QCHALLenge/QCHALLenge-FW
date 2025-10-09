import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering

from sp.data.sp_data import SPData


class SPDecomposer:
    def __init__(self, data, decomposition_type="clustering") -> None:
        self.data=data
        self.subproblems = []

        dec_funcs = {
            "clustering": self.decompose_clustering,
            "bisection": self.decompose_bisection,
            "vertical_cut": self.decompose_vert_cut
        }
        dec_funcs[decomposition_type]()

    def decompose_clustering(self, num_subproblems = 2):
        node_to_id = {node: idx for idx, node in enumerate(self.data.G.nodes())}
        id_to_node = {idx: node for node, idx in node_to_id.items()}

        sc = SpectralClustering(
            n_clusters=num_subproblems, 
            affinity='precomputed', 
            assign_labels='discretize',
            random_state=100
        )

        adjacency_matrix = nx.to_numpy_array(self.data.G, nodelist=node_to_id.keys())
        labels = sc.fit_predict(adjacency_matrix)

        subproblems = [set() for _ in range(num_subproblems)]
        for idx, label in enumerate(labels):
            node = id_to_node[idx]
            subproblems[label].add(node)

        for subproblem in subproblems:
            list_lidar = [l for l in subproblem if len(l) == 5]
            list_street_points = [sp for sp in subproblem if len(sp) == 3]
            problem_dict = {'listLidar':list_lidar, 'listCovering':list_street_points, 'wall': self.data.walls}
            self.subproblems.append(SPData.create_cls(problem_dict, self.data.data_params))

    def decompose_bisection(self):
        subproblems = nx.algorithms.community.kernighan_lin_bisection(self.data.G)
        subproblems = [set(subproblems[0]), set(subproblems[1])]

        for subproblem in subproblems:
            list_lidar = [l for l in subproblem if len(l) == 5]
            list_street_points = [sp for sp in subproblem if len(sp) == 3]
            problem_dict = {'listLidar':list_lidar, 'listCovering':list_street_points, 'wall': self.data.walls}
            self.subproblems.append(SPData.create_cls(problem_dict, self.data.data_params))

    def decompose_vert_cut(self):
        all_nodes = self.data.listLidar + self.data.listStreetPoints
        all_x_coordinates = [n[0] for n in all_nodes]
        avg_x = sum(all_x_coordinates) / len(all_x_coordinates)

        point_1 = [avg_x,0]
        point_2 = [avg_x,1]

        self.decompose_cut(point_1, point_2)

    def decompose_cut(self, point_1, point_2):

        separator = np.array([point_2[0] - point_1[0], point_2[1] - point_1[1]])
        perpendicular=np.array([-separator[1], separator[0]])

        list_lidar_1 = [s for s in self.data.listLidar if np.dot(np.array([s[0]-point_1[0], s[1]-point_1[1]]),perpendicular)>0]
        list_lidar_2 = [s for s in self.data.listLidar if np.dot(np.array([s[0]-point_1[0], s[1]-point_1[1]]),perpendicular)<=0]

        street_points_1 = [s for s in self.data.listStreetPoints if np.dot(np.array([s[0]-point_1[0], s[1]-point_1[1]]),perpendicular)>0]
        street_points_2 = [s for s in self.data.listStreetPoints if np.dot(np.array([s[0]-point_1[0], s[1]-point_1[1]]),perpendicular)<=0]

        problem_dict_1 = {'listLidar':list_lidar_1, 'listCovering':street_points_1, 'wall': self.data.walls}
        problem_dict_2 = {'listLidar':list_lidar_2, 'listCovering':street_points_2, 'wall': self.data.walls}

        self.subproblems.append(SPData.create_cls(problem_dict_1, self.data.data_params))
        self.subproblems.append(SPData.create_cls(problem_dict_2, self.data.data_params))
        
        
    def __compose_solutions(self, solution_dict1, solution_dict2):
        solution_dict = {**solution_dict1, **solution_dict2}
        for key, value in solution_dict.items():
            if key in solution_dict1 and key in solution_dict2:
                if solution_dict1[key]>solution_dict2[key]:
                    solution_dict[key] = solution_dict1[key]
                else: 
                    solution_dict[key] = solution_dict2[key]
        return solution_dict
    
    def solve(self, model_class, solve_func = None, **config):

        solutions = []
        for subproblem in self.subproblems:
            model = model_class(subproblem)
            if solve_func:
                answer = model.solve(solve_func, **config)
            else:
                answer = model.solve(**config)
            solutions.append(answer["solution"])
        
        global_solution = self.__compose_solutions(solutions[0], solutions[1])
        result = {"solution": global_solution, "energy": 0, "runtime": 0}

        return result

