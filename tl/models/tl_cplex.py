# remote import
from docplex.mp.model import Model
import sys
import os
from abstract.models.abstract_model import AbstractModel
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# local import
from tl.evaluation.tl_solution import Tl2D_Solution
from tl.models.tl_generic import Tl2D_Generic


class TLD2D_Cplex(Tl2D_Generic, AbstractModel):
    """
    This class implements the 2 dimensional Truckloading problem.
    """

    model_name = "Cplex"

    def __init__(self, data):
        super().__init__(data)
        self.build_model()

    def build_model(self):
        """
        Build and configure the optimization model.

        This method performs the following tasks:

        1. Generate Model Instance:
           - Initializes the model instance using the class attribute `self.model_name`.

        2. Define Binary Variables:
           - `b`: Binary variable dict to indicate whether each box is selected or not.
           - `r`: Binary variable dict to indicate whether each box is rotated or not.
           - `no_overlap_x`: Binary variable dict for enforcing no overlap in the x-direction between pairs of permuted boxes.
           - `no_overlap_y`: Binary variable dict for enforcing no overlap in the y-direction between pairs of permuted boxes.
           - These binary variables are stored as attributes of the class (`self.b`, `self.r`, `self.no_overlap_x`, `self.no_overlap_y`).

        3. Define Continuous Variables:
           - `x`: Continuous variable dict for the x-coordinate of the lower-left corner of each box.
           - `y`: Continuous variable dict for the y-coordinate of the lower-left corner of each box.
           - `x_length`: Continuous variable dict for the length of each box in the x-direction.
           - `y_length`: Continuous variable dict for the length of each box in the y-direction.
           - `xp`: Continuous variable dict for the x-coordinate of the lower-left corner after potential rotation.
           - `yp`: Continuous variable dict for the y-coordinate of the lower-left corner after potential rotation.
           - These continuous variables are stored as attributes of the class (`self.x`, `self.y`, `self.x_length`, `self.y_length`, `self.xp`, `self.yp`).

        4. Define Objective:
           - Calls `define_objective(m, b)` to set the objective function of the model, using the model instance `m` and the binary variable `b`.

        5. Store Model:
           - The constructed model is stored in the class attribute `self.model`.

        6. Add Constraints:
           - Capacity Constraint: Calls `constraint_capacity()` to add constraints related to the capacity of the container (or truck).
           - No Overlap Constraint: Calls `constraint_no_overlap(permut_boxes=permut_boxes)` to ensure that boxes do not overlap in the x and y directions.
           - Area Constraint: Calls `constraint_area()` to ensure that the boxes fit within the allowed area.
           - Rotation Constraint: Calls `constraint_rotation()` to allow or disallow rotation of the boxes.
           - Geometric Constraint: Calls `constraint_geometric()` to enforce any additional geometric constraints related to box placement.

        :return: None
        """

        # generate model instance
        m = Model(self.model_name)

        # define binary variables
        b = m.binary_var_dict(self.boxes, name="b")
        self.b = b
        r = m.binary_var_dict(self.boxes, name="r")
        self.r = r

        permut_boxes = self.permuted_box
        no_overlap_x = m.binary_var_dict(permut_boxes, name="no_overlap_x")
        self.no_overlap_x = no_overlap_x
        no_overlap_y = m.binary_var_dict(permut_boxes, name="no_overlap_y")
        self.no_overlap_y = no_overlap_y

        # define integer variables
        x = m.integer_var_dict(self.boxes, lb=0, name="x")
        self.x = x
        y = m.integer_var_dict(self.boxes, lb=0, name="y")
        self.y = y
        x_length = m.integer_var_dict(self.boxes, name="x_length")
        self.x_length = x_length
        y_length = m.integer_var_dict(self.boxes, name="y_length")
        self.y_length = y_length

        xp = m.integer_var_dict(self.boxes, lb=0, name="xp")
        self.xp = xp
        yp = m.integer_var_dict(self.boxes, lb=0, name="yp")
        self.yp = yp


        # define objective
        self.define_objective(m, b)

        self.model = m

        # add capacity constraint
        self.constraint_capacity()

        # forbid overlap
        self.constraint_no_overlap(permut_boxes=permut_boxes)

        # add area constraint
        self.constraint_area()

        # add rotation constraint
        self.constraint_rotation()

        # add geometric constraint
        self.constraint_geometric()

    def constraint_area(self):
        self.model.add_constraint(
            sum(self.data.boxes.loc[i]["area"] * self.b[i] for i in self.boxes)
            <= self.data.truck_length * self.data.truck_width
        )

        # add geometric constraints

    def constraint_geometric(self):

        self.model.add_constraints(
            (self.xp[i] <= self.data.truck_length for i in self.boxes)
        )
        self.model.add_constraints(
            (self.yp[i] <= self.data.truck_width for i in self.boxes)
        )

    def constraint_rotation(self):

        # add rotation constraints
        self.model.add_constraints(
            (
                self.x_length[i]
                == (1 - self.r[i]) * self.data.boxes.loc[i]["length"]
                + self.data.boxes.loc[i]["width"] * self.r[i]
                for i in self.boxes
            )
        )
        self.model.add_constraints(
            (self.xp[i] - self.x[i] == self.x_length[i] for i in self.boxes)
        )

        self.model.add_constraints(
            (
                self.y_length[i]
                == (1 - self.r[i]) * self.data.boxes.loc[i]["width"]
                + self.data.boxes.loc[i]["length"] * self.r[i]
                for i in self.boxes
            )
        )

        self.model.add_constraints(
            (self.yp[i] - self.y[i] == self.y_length[i] for i in self.boxes)
        )

    def constraint_no_overlap(self, permut_boxes):
        x = self.x
        xp = self.xp
        b = self.b
        y = self.y
        yp = self.yp
        no_overlap_x = self.no_overlap_x
        no_overlap_y = self.no_overlap_y

        # add non overlapping constraints
        self.model.add_constraints(
            (
                x[i] >= xp[j] - self.data.truck_length * (1 - no_overlap_x[(i, j)])
                for (i, j) in permut_boxes
            )
        )
        self.model.add_constraints(
            (
                y[i] >= yp[j] - self.data.truck_width * (1 - no_overlap_y[(i, j)])
                for (i, j) in permut_boxes
            )
        )

        self.model.add_constraints(
            (
                no_overlap_x[i, j]
                + no_overlap_x[j, i]
                + no_overlap_y[i, j]
                + no_overlap_y[j, i]
                >= b[i] + b[j] - 1
                for (i, j) in permut_boxes
            )
        )

    def constraint_capacity(self):
        self.model.add_constraint(
            sum(self.data.boxes.loc[i]["weight"] * self.b[i] for i in self.boxes)
            <= self.data.truck_capacity
        )

    def define_objective(self, model, b):
        model.objective_expr = sum(
            b[i] * self.data.boxes.loc[i]["area"] for i in b.keys()
        )
        model.objective_sense = "max"

    def set_time_limit(self, time_limit_sec: int):
        self.model.time_limit = time_limit_sec

    def solve(self, *args, **kwargs):
        self.optimize()

    def optimize(self, **config):
        """

        :return:
        """
        TimeLimit = config.get("TimeLimit", 10)
        self.set_time_limit(time_limit_sec=TimeLimit)
        self.model_solution = self.model.solve()

    def load_from_csv(self):
        """

        :return:
        """

    def toggel_output_logging(self, turn_off=True):
        """

        :param turn_off:
        :return:
        """
        return None

    def generate_solution_overview(self):
        """
        This function is used to standardize the model output and allow for plotting overview.
        :return:
        """
        runtime = self.model_solution.solve_details.time
        n_var = self.model.number_of_variables
        n_constr = self.model.number_of_constraints
        selected_boxes = {}
        for box, var in self.b.items():
            if self.model_solution[var.name] > 0.1:
                x = self.x[box]
                y = self.y[box]
                r = self.r[box]
                x_coord = self.model_solution[x.name]
                y_coord = self.model_solution[y.name]
                rotation = self.model_solution[r.name]
                selected_boxes[box] = {
                    "x_coord": x_coord,
                    "y_coord": y_coord,
                    "rotation": rotation,
                }

        self.solution = Tl2D_Solution(
            selected_boxes=selected_boxes,
            runtime=runtime,
            num_vars=n_var,
            num_constrs=n_constr,
            obj_val=self.model_solution.get_objective_value(),
            data=self.data,
            model_name=self.model_name,
            violated_constraints=[],
            gap_to_optimal=self.model_solution.solve_details.mip_relative_gap,
        )

        return self.solution

        # constraint checks

    def check_capacity_constraint(self, solution):
        data = self.data
        selected_boxes = self.select_boxes(solution)
        total_weight = sum(
            data.boxes.loc[i]["weight"] for i in selected_boxes.keys()
        )
        if total_weight <= data.truck_capacity:
            return []
        else:
            # Return all boxes as they collectively violate the constraint
            return list(selected_boxes.keys())

    def check_area_constraint(self, solution):
        data = self.data
        selected_boxes = self.select_boxes(solution)
        total_area = sum(
            data.boxes.loc[i]["area"] for i in selected_boxes.keys()
        )
        if total_area <= data.truck_length * data.truck_width:
            return []
        else:
            # Return all boxes as they collectively violate the constraint
            return list(selected_boxes.keys())

    def check_geometric_constraints(self, solution):
        data = self.data
        selected_boxes = self.select_boxes(solution)
        violations = []
        for box, values in selected_boxes.items():
            if (
                values["x_coord"] > data.truck_length
                or values["y_coord"] > data.truck_width
            ):
                violations.append(box)
        return violations

    def check_rotation_constraints(self, solution):
        data = self.data
        selected_boxes = self.select_boxes(solution)
        violations = []
        for box, values in selected_boxes.items():
            length = (
                data.boxes.loc[box]["length"]
                if values["rotation"] == 0
                else data.boxes.loc[box]["width"]
            )
            width = (
                data.boxes.loc[box]["width"]
                if values["rotation"] == 0
                else data.boxes.loc[box]["length"]
            )
            if (
                values["x_coord"] + length > data.truck_length
                or values["y_coord"] + width > data.truck_width
            ):
                violations.append(box)
        return violations

    def check_no_overlap_constraint(self, solution):
        data = self.data
        selected_boxes = self.select_boxes(solution)
        violations = set()

        for box1 in selected_boxes:
            for box2 in selected_boxes:
                if box1 == box2 or box1 in violations and box2 in violations:
                    continue

                x1, y1, r1 = (
                    round(selected_boxes[box1]["x_coord"]),
                     round(selected_boxes[box1]["y_coord"]),
                     round(selected_boxes[box1]["rotation"]),
                )
                x2, y2, r2 = (
                     round(selected_boxes[box2]["x_coord"]),
                     round(selected_boxes[box2]["y_coord"]),
                     round(selected_boxes[box2]["rotation"]),
                )
                w1 =  round((
                    data.boxes.loc[box1]["width"]
                    if r1 == 0
                    else data.boxes.loc[box1]["length"]
                ))
                l1 =  round((
                    data.boxes.loc[box1]["length"]
                    if r1 == 0
                    else data.boxes.loc[box1]["width"]
                ))
                w2 =  round((
                    data.boxes.loc[box2]["width"]
                    if r2 == 0
                    else data.boxes.loc[box2]["length"]
                ))
                l2 =  round((
                    data.boxes.loc[box2]["length"]
                    if r2 == 0
                    else data.boxes.loc[box2]["width"]
                ))

                if not (
                    x1 + l1 <= x2 or x2 + l2 <= x1 or y1 + w1 <= y2 or y2 + w2 <= y1
                ):
                    if (box2,box1) not in violations:
                        violations.add((box1, box2))

        return list(violations)

    def check_all_constraints(self, solution):
        return {
            "capacity": self.check_capacity_constraint(solution),
            "area": self.check_area_constraint(solution),
            "geometric": self.check_geometric_constraints(solution),
            "rotation": self.check_rotation_constraints(solution),
            "no_overlap": self.check_no_overlap_constraint(solution),
        }
    
    def select_boxes(self, solution):
        selected_boxes = {}
        for key, value in solution.items():
            if key.startswith("b_") and value != 0:
                box = int(key.split("_")[1])
                selected_boxes[box] = {
                    "x_coord": solution.get(f"x_{box}", 0),
                    "y_coord": solution.get(f"y_{box}", 0),
                    "rotation": solution.get(f"r_{box}", 0),
                }
        return selected_boxes