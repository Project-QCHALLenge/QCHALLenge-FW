# remote imports´
import gurobipy as gp
from gurobipy import GRB

import sys
import os
from abstract.models.abstract_model import AbstractModel
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
# local import
from tl.evaluation.tl_solution import Tl2D_Solution
from tl.models.tl_generic import Tl2D_Generic


class TLD2D_Gurobi(Tl2D_Generic, AbstractModel):
    """This functions implements an 2D case of the truck loading problem"""

    model: gp.Model
    model_name = "Gurobi"

    def __init__(self, data):
        super().__init__(data)
        self.model = self.build_model()

    def solve(self, **params):

        timelimit = params.get("TimeLimit", 1)
        self.model.Params.TimeLimit = timelimit

        self.model.optimize()
        if self.model.solCount == 0:
            raw_solution = {}
            optimality_gap = None
        else:
            raw_solution = {var.VarName: int(var.X) for var in self.model.getVars()}
            optimality_gap = self.model.MIPGap

        self.selected_boxes = {}
        for box, var in self.model._b.items():
            if var.X > 0.1:
                x_coord = self.model._x[box].X
                y_coord = self.model._y[box].X
                rotation = self.model._r[box].X
                self.selected_boxes[box] = {
                    "x_coord": x_coord,
                    "y_coord": y_coord,
                    "rotation": rotation,
                }

        info = self.generate_solution_overview()
        converted_solution = {}
        for key, value in raw_solution.items():
            new_key = key.replace("[", "_").replace("]", "").replace(",", "_")
            converted_solution[new_key] = value
        answer = {}
        answer["energy"] = None
        answer["runtime"] = self.model.RunTime
        answer["converted solution"] = converted_solution
        answer["optimality_gap"] = optimality_gap
        answer["solution"] = {"model_name":self.model_name, "num_vars":len(raw_solution), "solution":self.selected_boxes}
        answer["info"] = info.to_dict()

        # return answer
        return answer

    def build_model(self) -> gp.Model:
        """
        Generate the Gurobi model and define decision variables, objective function, and constraints.

        This function performs the following steps:

        1. Initialize Gurobi Model:
           - Creates a new Gurobi model with the name specified by `self.model_name`.
           - The model is stored as `self.model`.

        2. Define Binary Variables:
           - `u`: Binary variables for no overlap in the x-direction for permuted boxes.
           - `v`: Binary variables for no overlap in the y-direction for permuted boxes.
           - `b`: Binary variables to indicate whether each box is selected or not.
           - `r`: Binary variables to indicate whether each box is rotated or not.
           - These variables are stored as attributes of the model (`truck._u`, `truck._v`, `truck._b`, `truck._r`).

        3. Define Continuous Variables:
           - `x`: Continuous variables for the x-coordinate of the lower-left corner of each box.
           - `y`: Continuous variables for the y-coordinate of the lower-left corner of each box.
           - `x_length`: Continuous variables for the length of each box in the x-direction.
           - `y_length`: Continuous variables for the length of each box in the y-direction.
           - These variables are stored as attributes of the model (`truck._x`, `truck._y`, `truck._x_length`, `truck._y_length`).

        4. Define Objective:
           - Calls `self.define_objective(model=truck)` to set the objective of the model, which is to maximize the area used in the truck.

        5. Add Constraints:
           - Box Dimensions Constraint: Calls `self.constraint_determine_box_dimensions(truck, x_length, y_length, r, data)` to determine the dimensions of each box.
           - Area Constraint: Calls `self.constraint_area(truck, b, data)` to ensure that the area of selected boxes does not exceed the truck's volume.
           - Weight Constraint: Calls `self.constraint_weight()` to limit the weight of the selected boxes.
           - Neighbouring Box Constraint: Calls `self.constriant_limit_box_by_neighbouring_boxes(truck, x, x_length, u, y, y_length, v, data)` to ensure that the positions of the boxes depend on their neighbours.
           - Box Position Constraint: Calls `self.constraint_box_in_truck(truck, x, x_length, y, y_length, b, data)` to ensure that all boxes are positioned within the truck.
           - No Overlapping Constraint: Calls `self.constraint_no_overlapping(truck)` to ensure that boxes do not overlap within the truck.

        6. Finalize Model:
           - Calls `truck.update()` to finalize the model after adding all variables and constraints.

        :return: The Gurobi model instance (`truck`).
        """

        truck = gp.Model(self.model_name)
        self.model = truck
        self.model.Params.LogToConsole = 0
        self.model.Params.OutputFlag = 0
        data = self.data
        boxes = data.get_box_names_as_list()

        # define binary variables
        permuted_box = self.permuted_box
        u = truck.addVars(permuted_box, lb=0, ub=1, vtype=GRB.BINARY, name="u")
        truck._u = u
        v = truck.addVars(permuted_box, lb=0, ub=1, vtype=GRB.BINARY, name="v")
        truck._v = v
        b = truck.addVars(boxes, lb=0, vtype=GRB.BINARY, name="b")
        truck._b = b
        r = truck.addVars(boxes, lb=0, vtype=GRB.BINARY, name="r")
        truck._r = r

        # define continuous variables
        x = truck.addVars(boxes, vtype=GRB.CONTINUOUS, name="x")
        truck._x = x
        y = truck.addVars(boxes, vtype=GRB.CONTINUOUS, name="y")
        truck._y = y
        x_length = truck.addVars(boxes, vtype=GRB.CONTINUOUS, name="length_x")
        truck._x_length = x_length
        y_length = truck.addVars(boxes, vtype=GRB.CONTINUOUS, name="length_y")
        truck._y_length = y_length

        # we want to maximise the area used in the truck
        self.define_objective(model=truck)

        # define constraints to model
        self.constraint_determine_box_dimensions(truck, x_length, y_length, r, data)

        # limit the area of the selected boxes to be at most the truck volumne
        self.constraint_area(truck, b, data)

        # limit the weigth
        self.constraint_weight()

        # limit the position of the boxes in interdependancy
        self.constriant_limit_box_by_neighbouring_boxes(
            truck, x, x_length, u, y, y_length, v, data
        )

        # limit the box to be in the truck -> geometric constraint
        self.constraint_box_in_truck(truck, x, x_length, y, y_length, b, data)

        # box needs to be neighboured by another
        self.constraint_no_overlapping(truck)

        truck.update()

        return truck

    def constraint_weight(self):
        b = self.model._b
        self.model.addConstr(
            gp.quicksum(self.data.boxes.loc[i]["weight"] * b[i] for i in self.boxes)
            <= self.data.truck_capacity,
            name="weight_constraint",
        )

    def set_time_limit(self, time_limit_sec: int):  #
        self.model.setParam("TimeLimit", time_limit_sec)

    def optimize(self, **config):
        TimeLimit = config["TimeLimit"]
        self.set_time_limit(time_limit_sec=TimeLimit)
        self.model.optimize()

    def constraint_no_overlapping(self, model):
        b = model._b
        u = model._u
        v = model._v

        model.addConstrs(
            b[i] + b[j] - 1 <= u[i, j] + u[j, i] + v[i, j] + v[j, i]
            for (i, j) in u.keys()
        )

        # limit boxes to be within truck

    def constraint_box_in_truck(self, model, x, x_length, y, y_length, b, data):
        # define upper limit on box positioning in x axis
        model.addConstrs(
            x_length[i] + x[i] <= data.truck_length + (1 - b[i]) * data.truck_length
            for i in b.keys()
        )
        # daniel: this way we would not limit the x_i to 2 L even when the box is not picked
        # model.addConstrs(
        #     b[i] * (x_length[i] + x[i]) <= data.truck_length
        #     for i in b.keys()
        # )

        # define upper limit on box positioning in y axis
        model.addConstrs(
            y_length[i] + y[i] <= data.truck_width + (1 - b[i]) * data.truck_width
            for i in b.keys()
        )

    def constriant_limit_box_by_neighbouring_boxes(
        self, model, x, x_length, u, y, y_length, v, data
    ):

        model.addConstrs(
            x_length[i] + x[i] <= x[j] + (1 - u[i, j]) * data.truck_length
            for i, j in u.keys()
        )

        model.addConstrs(
            y_length[i] + y[i] <= y[j] + (1 - v[i, j]) * data.truck_length
            for i, j in v.keys()
        )

    def constraint_area(self, model, box_selection_var, data):
        model.addConstr(
            gp.quicksum(
                box_selection_var[box_index] * data.boxes.loc[box_index]["area"]
                for box_index in box_selection_var.keys()
            )
            <= data.truck_length * data.truck_width
        )

    def constraint_determine_box_dimensions(
        self, model, x_length_var, y_length_var, rotation_var, data
    ):
        model.addConstrs(
            x_length_var[box_index]
            == data.boxes.loc[box_index]["length"] * (1 - rotation_var[box_index])
            + rotation_var[box_index] * data.boxes.loc[box_index]["width"]
            for box_index in rotation_var.keys()
        )

        model.addConstrs(
            y_length_var[box_index]
            == data.boxes.loc[box_index]["width"] * (1 - rotation_var[box_index])
            + rotation_var[box_index] * data.boxes.loc[box_index]["length"]
            for box_index in rotation_var.keys()
        )

    def define_objective(self, model):
        # define objective
        obj = gp.quicksum(
            model._b[i] * self.data.boxes.loc[i]["area"] for i in model._b.keys()
        )
        model.setObjective(obj, GRB.MAXIMIZE)

    def toggel_output_logging(self, turn_off=True):
        if turn_off:
            self.model.setParam("OutputFlag", 0)
        else:
            self.model.setParam("OutputFlag", 1)

    def generate_solution_overview(self):
        """
        This function is used to standardize the model output and allow for plotting overview.
        :return:
        """
        if self.model.Status == GRB.OPTIMAL:
            runtime = self.model.Runtime
            n_var = self.model.NumVars
            n_constr = self.model.NumConstrs

            solution = Tl2D_Solution(
                model_name=self.model_name,
                selected_boxes=self.selected_boxes,
                runtime=runtime,
                num_vars=n_var,
                num_constrs=n_constr,
                obj_val=self.model.objVal,
                data=self.data,
                violated_constraints=[],
                gap_to_optimal=self.model.MIPGap,
                energy=0,
            )

            return solution

    # constraint checks

    def check_no_overlapping_violation(self, solution):
        model = self.model
        b = {i: solution[f"b[{i}]"] for i in model._b}
        u = {(i, j): solution[f"u[{i},{j}]"] for (i, j) in model._u}
        v = {(i, j): solution[f"v[{i},{j}]"] for (i, j) in model._v}
        violations = []
        for i, j in model._u.keys():
            if b[i] + b[j] - 1 > u[i, j] + u[j, i] + v[i, j] + v[j, i]:
                violations.append((i, j))
        return violations

    def check_weight_violation(self, solution):
        model = self.model
        b = {i: solution[f"b[{i}]"] for i in model._b}
        total_weight = sum(model.data.boxes.loc[i]["weight"] * b[i] for i in model._b)
        return total_weight > model.data.truck_capacity

    def check_box_in_truck_violation(self, solution):
        model = self.model
        violations = []
        b = {i: solution[f"b[{i}]"] for i in model._b}
        x = {i: solution[f"x[{i}]"] for i in model._x}
        y = {i: solution[f"y[{i}]"] for i in model._y}
        x_length = {i: solution[f"length_x[{i}]"] for i in model._x_length}
        y_length = {i: solution[f"length_y[{i}]"] for i in model._y_length}

        for i in model._b.keys():
            if b[i] and (
                x[i] + x_length[i] > model.data.truck_length
                or y[i] + y_length[i] > model.data.truck_width
            ):
                violations.append(i)
        return violations

    def check_neighbouring_boxes_violation(self, solution):
        model = self.model
        violations = []
        u = {(i, j): solution[f"u[{i},{j}]"] for (i, j) in model._u}
        v = {(i, j): solution[f"v[{i},{j}]"] for (i, j) in model._v}
        x = {i: solution[f"x[{i}]"] for i in model._x}
        y = {i: solution[f"y[{i}]"] for i in model._y}
        x_length = {i: solution[f"length_x[{i}]"] for i in model._x_length}
        y_length = {i: solution[f"length_y[{i}]"] for i in model._y_length}

        for i, j in model._u.keys():
            if x[i] + x_length[i] > x[j] + (1 - u[i, j]) * model.data.truck_length:
                violations.append((i, j, "x"))
            if y[i] + y_length[i] > y[j] + (1 - v[i, j]) * model.data.truck_length:
                violations.append((i, j, "y"))
        return violations

    def check_area_violation(self, solution):
        model = self.model
        b = {i: solution[f"b[{i}]"] for i in model._b}
        total_area = sum(b[i] * model.data.boxes.loc[i]["area"] for i in model._b)
        return total_area > model.data.truck_length * model.data.truck_width

    def check_box_dimension_violation(self, solution):
        model = self.model
        violations = []
        r = {i: solution[f"r[{i}]"] for i in model._r}
        x_length = {i: solution[f"length_x[{i}]"] for i in model._x_length}
        y_length = {i: solution[f"length_y[{i}]"] for i in model._y_length}

        for i in model._r.keys():
            expected_x_length = (
                model.data.boxes.loc[i]["length"] * (1 - r[i])
                + r[i] * model.data.boxes.loc[i]["width"]
            )
            expected_y_length = (
                model.data.boxes.loc[i]["width"] * (1 - r[i])
                + r[i] * model.data.boxes.loc[i]["length"]
            )

            if x_length[i] != expected_x_length or y_length[i] != expected_y_length:
                violations.append(i)
        return violations

    def check_all_constraints(self, solution):
        return {
            "capacity": self.check_weight_violation(solution),
            "area": self.check_area_violation(solution),
            "geometric": self.check_box_in_truck_violation(solution),
            "rotation": self.check_rotation_constraints(solution),
            "no_overlap": self.check_no_overlapping_violation(solution),
        }
