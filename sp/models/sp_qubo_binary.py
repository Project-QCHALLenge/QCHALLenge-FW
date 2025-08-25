import time
import math

import numpy as np


class QuboSPBinary:
    def __init__(self, data, P=2, W=0, distance = True, reduced=False, **params) -> None:
        self.data = data
        self.reduced = reduced

        if reduced:
            self.G = self.data.G_reduced
        else:
            self.G = self.data.G

        self.usedLidars = []
        self.mandatoryLidars = []

        self.custom_edge_weights = {}
        self.are_edges_weighted = distance
        self.calculate_custom_edge_weights()
        
        self.P = P
        self.W = W
        self.model = self.__build_model()

    def __inverter_solution(self, sample):
        solution_dict = {
            f"x_{self.usedLidars[i][0]}_{self.usedLidars[i][1]}_{self.usedLidars[i][2]}_{self.usedLidars[i][3]}_{self.usedLidars[i][4]}": 
            float(sample[i])
            for i in range(len(self.usedLidars))
        }
        if self.reduced:
            for l in self.data.lidar1:
                solution_dict[f"x_{l[0]}_{l[1]}_{l[2]}_{l[3]}_{l[4]}"] = 1
            for l in self.data.lidar0:
                solution_dict[f"x_{l[0]}_{l[1]}_{l[2]}_{l[3]}_{l[4]}"] = 0
        return solution_dict

    def solve(self, solve_func, **config):

        start_time = time.time()
        answer = solve_func(Q=self.model, **config)
        solve_time = time.time() - start_time

        solution = self.__inverter_solution(answer.first.sample)
        info = answer.info

        return {
            "solution": solution,
            "energy": answer.first.energy,
            "runtime": solve_time,
            "info": info,
        }

    def __needed_bitnum(self, decnum):
        if decnum == 0:
            return 0
        return int(math.ceil(math.log2(decnum)))

    def __is_in_list(self, mylist, target):
        for i in mylist:
            if target == i:
                return True
        return False
    
    def __weight_calculation(self, sx, sy, lx, ly):
        dist = math.sqrt((sx - lx) ** 2 + (sy - ly) ** 2)
        weight = self.data.max_radius - dist
        if weight < 0:
            weight = 0
        return weight
    
    def calculate_custom_edge_weights(self):

        for s in self.data.listStreetPoints:
            for l in self.data.listLidar:
                line = (l, s)
                if (self.data._in_range(l, s, self.data.max_radius, self.data.max_vert_angle, self.data.min_vert_angle, self.data.half_horizontal_angle)):
                    inters = 0
                    for w in self.data.walls:
                        inters += self.data._intersect(line, w)
                        if inters:
                            break
                    if inters == 0 and self.are_edges_weighted:
                        # We measure the distance of the nodes between each other and then calculate the
                        # edge weight as weight = max_radius - distance, if distance is greater than max_radius the
                        # weight is set to 0
                        weight = self.__weight_calculation(s[0], s[1], l[0], l[1])
                        key = frozenset([s, l])
                        self.custom_edge_weights[key] = weight

        values = list(self.custom_edge_weights.values())
        if len(values) == 0:
            self.custom_edge_weights = {}
        elif max(values) != min(values):
            self.custom_edge_weights = {
                key: (value - min(values)) / (max(values) - min(values))
                for key, value in self.custom_edge_weights.items()
            }
        else: 
            self.custom_edge_weights = {
                key: 1
                for key, value in self.custom_edge_weights.items()
            }
        

    def create_slacklist(self):
        slacksize = 0
        slacklist = []

        for node in self.G.nodes:
            # street points have len 3
            if len(node) == 3:
                adjacent_lidars = self.G.adj[node].items()
                slackbits = self.__needed_bitnum(len(adjacent_lidars))

                lidar_per_sp = []
                for lidar in adjacent_lidars:
                    # Collect lidar for the street point
                    self.usedLidars.append(lidar[0])
                    lidar_per_sp.append(lidar[0])
                    if slackbits == 0:
                        self.mandatoryLidars.append(lidar[0])

                slack_mapping = {slacksize + i + 1: 2**i for i in range(slackbits)}
                slacklist.append([lidar_per_sp, slack_mapping])
                slacksize += slackbits

        # Remove double lidars
        self.usedLidars = list(set(self.usedLidars))
        self.mandatoryLidars = list(set(self.mandatoryLidars))
        lidar_indices = {lidar: idx for idx, lidar in enumerate(self.usedLidars)}

        # Reindex slack variables and subtract:
        for s in slacklist:
            if s[1]:
                s[1] = {
                    key + len(self.usedLidars) - 1: -value
                    for key, value in s[1].items()
                }


        return slacklist, slacksize, lidar_indices

    def __build_model(self):

        slacklist, slacksize, lidar_indices = self.create_slacklist()

        num_qubits = len(self.usedLidars) + slacksize
        Qubo = np.zeros([num_qubits, num_qubits], dtype=float)

        # Objective
        for i in range(0, len(self.usedLidars)):
            Qubo[i, i] = 1

        # Penalty P2, if mandatory Lidar is not used
        for i in range(0, len(self.usedLidars)):
            if self.__is_in_list(self.mandatoryLidars, self.usedLidars[i]):
                Qubo[i, i] -= 2

        # Cover all streetpoints
        for slack in slacklist:
            if slack[1]:
                lidar_dict = {}
                for l in slack[0]:
                    lidar_dict[lidar_indices[l]] = 1
                lidar_dict.update(slack[1])

                for i in lidar_dict:
                    Qubo[i, i] -= 2 * self.P * lidar_dict[i]
                    for j in lidar_dict:
                        Qubo[i, j] += self.P * lidar_dict[i] * lidar_dict[j]

        # penalties for close lidars
        for i in range(len(self.usedLidars)):
            for j in range(len(self.usedLidars)):
                for s in self.G.nodes:
                    if len(s) == 3:
                        v_si = self.custom_edge_weights.get(frozenset([s, self.usedLidars[i]]), 0)
                        v_sj = self.custom_edge_weights.get(frozenset([s, self.usedLidars[j]]), 0)
                        # Penalty if close lidars are activated
                        Qubo[i, j] += self.W * v_sj * v_si

        return Qubo
