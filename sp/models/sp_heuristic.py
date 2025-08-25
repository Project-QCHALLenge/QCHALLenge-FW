import time

class SPHeuristic:

    def __init__(self, data):
        self.data = data

        listStreetPointsCovered = [x for x in self.data.listStreetPoints if x not in self.data.listStreetPointsNeverCovered]

        # create helpers
        self.index_to_sp = {i: lidar for i, lidar in enumerate(listStreetPointsCovered)}
        self.index_to_lidar = {i: lidar for i, lidar in enumerate(self.data.listLidar)}
        self.sp_to_index = {lidar: i for i, lidar in enumerate(listStreetPointsCovered)}
        self.lidar_to_index = {lidar: i for i, lidar in enumerate(self.data.listLidar)}

        self.lidar = list(self.index_to_lidar.keys())
        self.sp = list(self.index_to_sp.keys())

        self.lidar_covers = []
        for lidar in self.data.listLidar:
            sp_covered = [self.sp_to_index[sp] for sp in list(self.data.G.neighbors(lidar))]
            self.lidar_covers.append(sp_covered)

        self.sp_num_covered = [0 for i in self.sp]
        for lidar in self.lidar:
            for sp in self.lidar_covers[lidar]:
                self.sp_num_covered[sp] += 1

        self.solution = [None for _ in self.lidar]

    def activate_lidar(self, lidar):
        self.solution[lidar] = 1
        self.lidar.remove(lidar)
        for sp in self.lidar_covers[lidar]:
            if sp in self.sp:
                self.sp.remove(sp)
            self.sp_num_covered[sp] = None
        
    def set_mandatory(self):
        mandatory_list = []
        for sp in self.sp:
            if self.sp_num_covered[sp] == 1:
                for lidar in self.lidar:
                    if sp in self.lidar_covers[lidar]:
                        mandatory_list.append(lidar)
        mandatory_list = list(set(mandatory_list))
        for lidar in mandatory_list:
            self.activate_lidar(lidar)

    def deactivate_lidar(self, deactivate_lidar):
        self.solution[deactivate_lidar] = 0
        self.lidar.remove(deactivate_lidar)
        for sp in self.lidar_covers[deactivate_lidar]:
            if self.sp_num_covered[sp] is not None:
                self.sp_num_covered[sp] -= 1

    def get_min_covered(self):
        min_val = float('inf')
        for val in self.sp_num_covered:
            if val is not None:
                if val < min_val:
                    min_val = val
        return min_val
    
    def get_max_covered(self):
        max_val = float('-inf')
        for val in self.sp_num_covered:
            if val is not None:
                if val > max_val:
                    max_val = val
        return max_val
    
    def calc_lidar_prios(self):
        min_covered = self.get_min_covered()
        max_covered = self.get_max_covered() + 1
        if max_covered == min_covered:
            sp_covered_prios = [0 if x is None else 1 for x in self.sp_num_covered]
        else:
            sp_covered_prios = [0 if x is None else (max_covered - x)/(max_covered-min_covered) for x in self.sp_num_covered]
        lidar_prios = []
        for lidar in self.lidar:
            prio = 0
            for covers in self.lidar_covers[lidar]:
                prio += sp_covered_prios[covers]
            lidar_prios.append(prio)
        return lidar_prios
    
    def remove_lidar(self):
        lidar_prios = self.calc_lidar_prios()
        deactivate_lidar_index = lidar_prios.index(min(lidar_prios))
        deactivate_lidar_num = self.lidar[deactivate_lidar_index]
        self.deactivate_lidar(deactivate_lidar_num)

    def add_lidar(self):
        lidar_prios = self.calc_lidar_prios()
        deactivate_lidar_index = lidar_prios.index(max(lidar_prios))
        deactivate_lidar_num = self.lidar[deactivate_lidar_index]
        self.activate_lidar(deactivate_lidar_num)

    def run(self, max_iterations, add_lidar_each_step=True):
        iteration = 0
        while True:
            iteration += 1
            self.set_mandatory()
            if not self.sp:
                break

            self.remove_lidar()
            if add_lidar_each_step:
                if not self.sp:
                    break

                self.add_lidar()
            if not self.sp or iteration == max_iterations:
                break

    def create_solution_dict(self):
        self.solution = [0 if x is None else x for x in self.solution]
        solution_dict = {
            f"x_{self.index_to_lidar[i][0]}_{self.index_to_lidar[i][1]}_{self.index_to_lidar[i][2]}_{self.index_to_lidar[i][3]}_{self.index_to_lidar[i][4]}": self.solution[i]
            for i in range(len(self.solution))
        }
        return solution_dict

    def solve(self, **config):
        add_lidar_each_step = config.get("optimized", True)
        start_time = time.time()
        self.run(max_iterations=-1, add_lidar_each_step=add_lidar_each_step)
        runtime = time.time() - start_time  
        solution_dict = self.create_solution_dict()
        return {
            "solution": solution_dict,
            "energy": 0,
            "runtime": runtime,
        }

        


