import gurobipy as gp
import pygcgopt as gcg
import pyscipopt as scip

import sys
import os
from abstract.models.abstract_model import AbstractModel
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from tl.evaluation.tl_solution import Tl2D_Solution
from .tl_generic import Tl2D_Generic


class TLD2D_Scip(Tl2D_Generic, AbstractModel):
    model_name = "Scip"

    def __init__(self, data):
        super().__init__(data)
        self.build_model()
        self.toggel_output_logging(True)

    def define_model_type(self):
        truck = scip.Model(enablepricing=True)

        return truck

    def build_model(self):
        """
        This function generates a truck loading problem model using the Dantzig-Wolfe decomposition approach
        in the context of a mixed-integer programming model. The model is built to optimize the placement
        of boxes within a truck subject to various constraints, maximizing the utilization of space.

        The following steps are performed:

        1. Initialize Model:
           - The model is initialized by calling `self.define_model_type()`, which determines the type of
             optimization model (e.g., linear, mixed-integer) to be used.
           - The objective is set to maximize the space utilization within the truck using `truck.setMaximize()`.
           - The model instance is stored as `self.model`.

        2. Define Binary Variables:
           - `u[i, j]`: Binary variables representing whether box `i` does not overlap with box `j` in the x-direction.
           - `v[i, j]`: Binary variables representing whether box `i` does not overlap with box `j` in the y-direction.
           - `b[i]`: Binary variables representing whether box `i` is selected to be placed in the truck.
             The objective coefficient is set to the area of the box.
           - `r[i]`: Binary variables representing whether box `i` is rotated.
           - These variables are stored in dictionaries `self._u`, `self._v`, `self._b`, and `self._r` respectively.

        3. Define Continuous Variables:
           - `x[i]`: Continuous variables representing the x-coordinate of the lower-left corner of box `i`.
           - `x_length[i]`: Continuous variables representing the length of box `i` in the x-direction.
           - `y[i]`: Continuous variables representing the y-coordinate of the lower-left corner of box `i`.
           - `y_length[i]`: Continuous variables representing the length of box `i` in the y-direction.
           - These variables are stored in dictionaries `self._x`, `self._x_length`, `self._y`, and `self._y_length` respectively.

        4. Add Constraints:
           - `constraint_volume()`: Ensures the total volume of the selected boxes does not exceed the truck's capacity.
           - `constraint_weight()`: Ensures the total weight of the selected boxes does not exceed the truck's weight limit.
           - `constraint_limit_box_by_neighbouring_boxes()`: Limits the position of boxes based on neighboring boxes.
           - `constraint_no_overlapping()`: Ensures that no two boxes overlap within the truck.
           - `constraint_determine_box_dimensions()`: Determines the dimensions of each box based on its orientation (rotated or not).
           - `constraint_box_in_truck()`: Ensures that all boxes are fully contained within the truck's dimensions.

        :return: None
        """

        truck = self.define_model_type()
        # define objective
        truck.setMaximize()

        self.model = truck

        # define binary variables
        u = {}
        self._u = u
        v = {}
        self._v = v
        for i, j in self.permuted_box:
            u[i, j] = truck.addVar(vtype="B", name=f"u_{i}_{j}")
            v[i, j] = truck.addVar(vtype="B", name=f"v_{i}_{j}")

        b = {}
        self._b = b
        r = {}
        self._r = r

        x = {}
        self._x = x
        x_length = {}
        self._x_length = x_length
        y = {}
        self._y = y
        y_length = {}
        self._y_length = y_length
        boxes_df = self.data.boxes

        for i in self.boxes:
            b[i] = truck.addVar(vtype="B", name=f"b_{i}", obj=boxes_df.loc[i]["area"])
            r[i] = truck.addVar(vtype="B", name=f"r_{i}")
            x[i] = truck.addVar(vtype="C", name=f"x_{i}")
            x_length[i] = truck.addVar(vtype="C", name=f"x_length_{i}")
            y[i] = truck.addVar(vtype="C", name=f"y_{i}")
            y_length[i] = truck.addVar(vtype="C", name=f"y_length{i}")

        # define box positioning in x-direction
        self.constraint_volume()
        self.constraint_weight()
        self.constraint_limit_box_by_neighbouring_boxes()
        self.constraint_no_overlapping()

        self.constraint_determine_box_dimensions()
        self.constraint_box_in_truck()

    def constraint_weight(self):
        b = self._b
        self._constr_weight = self.model.addCons(
            scip.quicksum(self.data.boxes.loc[i]["weight"] * b[i] for i in self.boxes)
            <= self.data.truck_capacity
        )

    def constraint_volume(self):
        self._constr_volume = self.model.addCons(
            gcg.quicksum(
                self._b[box_index] * self.data.boxes.loc[box_index]["area"]
                for box_index in self.boxes
            )
            <= self.data.truck_length * self.data.truck_width
        )

    def constraint_limit_box_by_neighbouring_boxes(self):
        self._constr_y_sequence = gp.tupledict()
        self._constr_x_sequence = gp.tupledict()
        for i, j in self.permuted_box:
            # define box poistioning in y direction
            self._constr_y_sequence[i, j] = self.model.addCons(
                self._x[i] + self._x_length[i]
                <= self._x[j] + (1 - self._u[i, j]) * self.data.truck_length
            )

            # define box positioning in y-direction
            self._constr_x_sequence[i, j] = self.model.addCons(
                self._y[i] + self._y_length[i]
                <= self._y[j] + (1 - self._v[i, j]) * self.data.truck_width
            )

    def constraint_no_overlapping(self):
        self._constr_neighbouring = gp.tupledict()
        for i, j in self.permuted_box:
            # selecting a neighbouring condition of two boxes are selected
            self._constr_neighbouring[i, j] = self.model.addCons(
                self._b[i] + self._b[j] - 1
                <= self._u[i, j] + self._u[j, i] + self._v[i, j] + self._v[j, i]
            )

    def constraint_determine_box_dimensions(self):
        self._constr_rotation_length = gp.tupledict()
        self._constr_rotation_width = gp.tupledict()

        for i in self.boxes:

            # define constraints
            self._constr_rotation_length[i] = self.model.addCons(
                self._x_length[i]
                == self.data.boxes.loc[i]["length"] * (1 - self._r[i])
                + self._r[i] * self.data.boxes.loc[i]["width"]
            )
            self._constr_rotation_width[i] = self.model.addCons(
                self._y_length[i]
                == self.data.boxes.loc[i]["width"] * (1 - self._r[i])
                + self._r[i] * self.data.boxes.loc[i]["length"]
            )

    def constraint_box_in_truck(self):
        self._constr_max_x = gp.tupledict()
        self._constr_max_y = gp.tupledict()

        for i in self.boxes:
            # define upper limit on box positioning in x axis
            self._constr_max_x[i] = self.model.addCons(
                self._x[i] + self._x_length[i] <= self.data.truck_length
            )

            # define upper limit on box positioning in x axis
            self._constr_max_y[i] = self.model.addCons(
                self._y[i] + self._y_length[i] <= self.data.truck_width
            )

    def set_time_limit(self, time_limit_sec):
        self.model.setParam("limits/time", time_limit_sec)

    def solve(self, **config):
        self.optimize(**config)

    def optimize(self, **config):
        if "TimeLimit" in config:
            TimeLimit = config.pop("TimeLimit")
            self.set_time_limit(TimeLimit)
        # self.model.solveConcurrent()
        self.model.optimize()

    def toggel_output_logging(self, turn_off: bool = True):
        if turn_off:
            self.model.setParam("display/verblevel", 0)
        else:
            self.model.setParam("display/verblevel", 1)

    def generate_solution_overview(self):
        selected_boxes = {}
        # Iterate over each box to plot it
        for box in self.boxes:
            box_value = self.model.getVal(self._b[box])
            if box_value > 0.5:  # Check if the box is placed
                x = self._x[box]
                x_coord = self.model.getVal(x)

                y = self._y[box]
                y_coord = self.model.getVal(y)
                r = self._r[box]

                r_value = self.model.getVal(r)

                selected_boxes[box] = {
                    "x_coord": x_coord,
                    "y_coord": y_coord,
                    "rotation": r_value,
                }

        runtime = self.model.getSolvingTime()

        n_var = self.model.getNVars()
        n_constr = self.model.getNConss()

        self.solution = Tl2D_Solution(
            selected_boxes=selected_boxes,
            runtime=runtime,
            num_vars=n_var,
            num_constrs=n_constr,
            obj_val=self.model.getObjVal(),
            data=self.data,
            model_name=self.model_name,
            violated_constraints=[],
            gap_to_optimal=self.model.getGap(),
        )
        print(
            f"With {self.model_name} found {self.solution.objective_value} in {self.solution.runtime:.1f} seconds"
        )
        return self.solution


class Tl2D_DantzigWolfe_Scip(TLD2D_Scip):
    model_name = "ColumnGeneration"

    def __init__(self, data):
        super().__init__(data)
        self.build_model()

    def define_model_type(self):
        self.model_name = "DantzigWolfe"
        truck = gcg.Model()
        truck.redirectOutput()

        return truck

    def optimize(self, **config):
        if "TimeLimit" in config:
            TimeLimit = config.pop("TimeLimit")
            self.set_time_limit(TimeLimit)
        # self.model.solveConcurrent()
        self.model.optimize()

    def build_model(self):
        super().build_model()
        self.define_box_decomposition()

    def define_box_decomposition(self):
        pd = self.model.createDecomposition()

        # define master constraints
        master_constraints = [
            *self._constr_y_sequence.values(),
            *self._constr_neighbouring.values(),
            *self._constr_x_sequence.values(),
            # *self._constr_max_x.values(),
            # *self._constr_max_y.values(),
            self._constr_volume,
            self._constr_weight,
        ]
        pd.fixConssToMaster(master_constraints)

        # für jede sag welche constraints in die subprlblem e
        for box in self.boxes:
            rotation_length = self._constr_rotation_length[box]
            rotation_width = self._constr_rotation_width[box]
            max_y = self._constr_max_y[box]
            max_x = self._constr_max_x[box]

            pd.fixConssToBlock([rotation_width, rotation_length, max_y, max_x], box)

        self.model.addPreexistingPartialDecomposition(pd)
        # decomp = self.model.listDecompositions()

        # tranlate
