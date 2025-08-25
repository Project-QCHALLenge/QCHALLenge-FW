from tl.models.tl_cplex import TLD2D_Cplex

class TLEvaluation:
    def __init__(self, data, solution):
        self.data = data
        self.solution_object = solution
        self.solution =  solution["solution"]
        self.model = TLD2D_Cplex(self.data)

    def check_solution(self, verbose: bool = False):
        return self.model.check_all_constraints(solution=self.solution_object)

    def get_objective(self) -> float:
        infeasible = 0
        selected_boxes = self.solution["solution"]
        for v in self.check_solution().values():
            infeasible += len(v)

        objective = 0
        for box_idx in selected_boxes.keys():
            objective += self.data.boxes.iloc[box_idx].area

        if infeasible:
            return -1
        else:
            return int(objective)

