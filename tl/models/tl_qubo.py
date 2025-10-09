# external imports
import time

import dimod
from dimod.binary import Binary

from transformations.from_cplex import FromCQM

import sys
import os
from abstract.models.abstract_model import AbstractModel
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from tl.evaluation.tl_solution import *
from tl.models.tl_generic import Tl2D_Generic


class TL2D_Qubo(Tl2D_Generic, AbstractModel):
    model_name = "Qubo"

    def __init__(self, data):
            super().__init__(data)
            self.build_model()
            
    def build_model(self):
        """
        This functions builts a dimod cqm model based on non-standard bin packing formulation. The formulation deviates
        from the standard by considering a variable for each box and step in each directions, i.e. a variable for
        each box and each possible step in the x direction. While this will yield significantly more variables, the
        corresponding constraints are tighter and thus can be formulated better in the qubo. For example x + z y <= 1,
        which can be represented as x*y*penalty in the final qubo.
        :return:
        """

        # each unit increase in the truck length and truck widht is considered a step, as this is also the smallest
        # unit for a box, thus num varialbes = dimension
        num_x_variables = self.data.truck_length
        num_y_variables = self.data.truck_width

        self.num_x_variables = num_x_variables
        self.num_y_variables = num_y_variables

        # we can fix variables under certain conditions, e.g. for a squared box the rotation can be fixed to 0
        self.fixed_varialbes = []

        # each box can at most be placed at a steps, where the remaining box can fit in the truck. Thus, we do not need
        # to consider variable combinations where the box would be outside the truck. Additionally, the box can be
        # rotated, thus we need to account for the minimum (= box width) and forbid the remainng overlap in a constraint.
        self.num_steps_x_by_box = {
            idx: int(num_x_variables - self.data.boxes.iloc[idx]["width"])
            for idx in self.boxes
        }

        self.num_steps_y_by_box = {
            idx: int(num_y_variables - self.data.boxes.iloc[idx]["width"])
            for idx in self.boxes
        }

        # we generate a variable for each box, for each step in each direction as well an indicator for the rotation
        self.x = {
            (box, x_step): Binary(label=f"x_{box}_{x_step}")
            for box in self.boxes
            for x_step in range(self.num_steps_x_by_box[box])
        }
        self.y = {
            (box, y_step): Binary(label=f"y_{box}_{y_step}")
            for box in self.boxes
            for y_step in range(self.num_steps_y_by_box[box])
        }

        self.r = {box: Binary(label=f"r_{box}") for box in self.boxes}

        # the penaly term is used to punsih any violations of the constraints
        penalty = 10
        self.penalty = penalty

        # instanciate the dimod cqm model
        model = dimod.ConstrainedQuadraticModel()
        self.model = model

        # define the required constraints for the 2d truck loading problem

        # each box can be placed once in each dimension
        self.constraint_box_at_most_once()

        # placed boxes are not allowed to overlapp with each other
        self.constraint_non_overlapping()

        # the box needs to be placed with in the truck
        self.constraint_box_inside_truck()

        # each step in each direction can have at most one box assigned
        self.constraint_only_one_per_segment()

        # add objective to the cqm model
        self.define_objective()

    def constraint_box_inside_truck(self):
        """
        See the mathematical notation in the readme for clarification and guidance.
        :return:
        """

        for i in self.boxes:
            diff_length_width = (
                self.data.boxes.iloc[i]["length"] - self.data.boxes.iloc[i]["width"]
            )

            # only need to perform rotation, if box is not square
            if diff_length_width > 0 and self.data.boxes.iloc[i]["length"] >= self.data.truck_parameters.truck_width and self.data.boxes.iloc[i]["width"] >= self.data.truck_parameters.truck_length:

                # limit box to be inside truck in x-axis
                max_u = self.num_steps_x_by_box[i]

                lhs = dimod.quicksum(
                    self.x[i, u] for u in range(max_u - diff_length_width, max_u)
                )
                rhs = self.r[i]

                self.model.add_constraint(
                    lhs - rhs <= 0,
                    penalty=self.penalty,
                    label=f"limit_box_in_truck_x_axis_{i}]",
                )

                # limit box to be inside truck in y axis
                max_v = self.num_steps_y_by_box[i]

                lhs = dimod.quicksum(
                    self.y[i, v] for v in range(max_v - diff_length_width, max_v)
                )
                rhs = 1 - self.r[i]

                self.model.add_constraint(
                    lhs - rhs <= 0,
                    label=f"limit_box_in_truck_y_axis_{i}",
                )
            else:
                # as the box is a square, we can fix the rotation variable
                self.fixed_varialbes.append(f"r_{i}")

    def define_objective(self):
        x = self.x
        y = self.y
        self.model.set_objective(
            dimod.quicksum(
                x[box, x_step] * y[box, y_step] * (-self.data.boxes.iloc[box]["area"])
                for box in self.boxes
                for y_step in range(self.num_steps_y_by_box[box])
                for x_step in range(self.num_steps_x_by_box[box])
            )
        )

    def set_time_limit(self, time_limit_sec):
        print("Time limit function not available for now")
        return None

    def constraint_box_at_most_once(self):
        """
        Each box can be placed at most one time in each dimension.
        :return:
        """
        cqm = self.model
        x = self.x
        y = self.y
        for box in self.boxes:

            # in x axis
            lhs = dimod.quicksum(
                x[box, x_step] for x_step in range(self.num_steps_x_by_box[box])
            )
            cqm.add_constraint(
                lhs <= 1,
                penalty=self.penalty,
                label=f"BoxOnlyOnceInX_{box}",
            )

            # in y axis
            lhs = dimod.quicksum(
                y[box, y_step] for y_step in range(self.num_steps_y_by_box[box])
            )
            cqm.add_constraint(
                lhs <= 1,
                penalty=self.penalty,
                label=f"BoyOnlyOnceInY_{box}",
            )

    def constraint_only_one_per_segment(self):

        x = self.x
        y = self.y
        cqm = self.model
        for x_step in range(self.num_x_variables):
            lhs = dimod.quicksum(
                x[box, x_step]
                for box in self.boxes
                if x_step < self.num_steps_x_by_box[box]
            )

            if lhs.num_variables >= 2:
                cqm.add_constraint(
                    lhs <= 1,
                    penalty=self.penalty,
                    label=f"AtMostOnePerSegemnt_x_{x_step}",
                )

        for y_step in range(self.num_y_variables):
            lhs = dimod.quicksum(
                y[box, y_step]
                for box in self.boxes
                if y_step < self.num_steps_y_by_box[box]
            )
            # we only need constraint if at least two boxes can be assigned to on step

            if lhs.num_variables >= 2:
                cqm.add_constraint(
                    lhs <= 1,
                    penalty=self.penalty,
                    label=f"AtMostOnePerSegemnt_y_{y_step}",
                )

    def constraint_non_overlapping(self):
        # no box can occupy neighbouring
        cqm = self.model
        x = self.x
        y = self.y

        for box_1 in self.boxes:
            width_steps = self.data.boxes.iloc[box_1]["width"]
            length_steps = self.data.boxes.iloc[box_1]["length"]
            length_diff = (
                self.data.boxes.iloc[box_1]["length"]
                - self.data.boxes.iloc[box_1]["width"]
            )

            # the non-overlapping constraint can be modified, if the box is square. This is due to the fact, that
            # the rotation can be ignored for a squared box

            if length_diff == 0:
                # x axis
                for x_step in range(self.num_steps_x_by_box[box_1]):
                    upper_step = min(self.num_x_variables, x_step + length_steps)

                    lhs = dimod.quicksum(
                        x[box_2, x_step_2]
                        for x_step_2 in range(x_step, upper_step)
                        for box_2 in self.boxes
                        if box_2 != box_1 and x_step_2 < self.num_steps_x_by_box[box_2]
                    )

                    if lhs.num_variables >= 1:
                        rhs = (1 - x[box_1, x_step]) * self.data.boxes.iloc[box_1][
                            "length"
                        ]

                        cqm.add_constraint(
                            lhs - rhs <= 0,
                            penalty=self.penalty,
                            label=f"NonOverlappingX_Square_{box_1}_{x_step}",
                        )

                # y axis
                for y_step in range(self.num_steps_y_by_box[box_1]):
                    upper_step = min(self.num_y_variables, y_step + width_steps)
                    lhs = dimod.quicksum(
                        y[box_2, y_step_2]
                        for y_step_2 in range(y_step, upper_step)
                        for box_2 in self.boxes
                        if box_2 != box_1 and y_step_2 < self.num_steps_y_by_box[box_2]
                    )
                    if lhs.num_variables >= 1:
                        rhs = (1 - y[box_1, y_step]) * self.data.boxes.iloc[box_1][
                            "length"
                        ]

                        cqm.add_constraint(
                            lhs - rhs <= 0,
                            penalty=self.penalty,
                            label=f"NonOverlappingY_Square_{box_1}_{y_step}",
                        )

            else:
                for x_step in range(self.num_steps_x_by_box[box_1]):

                    # no rotation takes place
                    upper_step = min(self.num_x_variables, x_step + length_steps)
                    length_diff = (
                        self.data.boxes.iloc[box_1]["length"]
                        - self.data.boxes.iloc[box_1]["width"]
                    )

                    lhs = dimod.quicksum(
                        x[box_2, x_step_2]
                        for x_step_2 in range(x_step, upper_step)
                        for box_2 in self.boxes
                        if box_2 != box_1 and x_step_2 < self.num_steps_x_by_box[box_2]
                    )

                    rhs = (1 - x[box_1, x_step]) * self.data.boxes.iloc[box_1][
                        "length"
                    ] + self.r[box_1] * length_diff
                    if lhs.num_variables >= 1:
                        cqm.add_constraint(
                            lhs - rhs <= 0,
                            penalty=self.penalty,
                            label=f"NonOverlappingX_NoRotation_{box_1}_{x_step}",
                        )

                    # rotation takes place in x dimension
                    upper_step = min(self.num_x_variables, x_step + width_steps)

                    lhs = dimod.quicksum(
                        x[box_2, x_step_2]
                        for x_step_2 in range(x_step, upper_step)
                        for box_2 in self.boxes
                        if box_2 != box_1 and x_step_2 < self.num_steps_x_by_box[box_2]
                    )
                    if lhs.num_variables >= 1:
                        rhs = (1 - x[box_1, x_step]) * self.data.boxes.iloc[box_1][
                            "width"
                        ]

                        cqm.add_constraint(
                            lhs - rhs <= 0,
                            penalty=self.penalty,
                            label=f"NonOverlappingX_WithRotation_{box_1}_{x_step}",
                        )

                for y_step in range(self.num_steps_y_by_box[box_1]):

                    # no rotation takes place
                    length_steps = int(self.data.boxes.iloc[box_1]["length"])
                    upper_step = min(self.num_y_variables, y_step + length_steps)
                    length_diff = (
                        self.data.boxes.iloc[box_1]["length"]
                        - self.data.boxes.iloc[box_1]["width"]
                    )

                    lhs = dimod.quicksum(
                        y[box_2, y_step_2]
                        for y_step_2 in range(y_step, upper_step)
                        for box_2 in self.boxes
                        if box_2 != box_1 and y_step_2 < self.num_steps_y_by_box[box_2]
                    )

                    if lhs.num_variables >= 1:
                        rhs = (1 - y[box_1, y_step]) * self.data.boxes.iloc[box_1][
                            "length"
                        ] + (1 - self.r[box_1]) * length_diff

                        cqm.add_constraint(
                            lhs - rhs <= 0,
                            penalty=self.penalty,
                            label=f"NonOverlappingY_NoRotation_{box_1}_{y_step}",
                        )

                    # no rotation takes place
                    width_steps = self.data.boxes.iloc[box_1]["width"]
                    upper_step = min(self.num_y_variables, y_step + width_steps)

                    lhs = dimod.quicksum(
                        y[box_2, y_step_2]
                        for y_step_2 in range(y_step, upper_step)
                        for box_2 in self.boxes
                        if box_2 != box_1 and y_step_2 < self.num_steps_y_by_box[box_2]
                    )
                    if lhs.num_variables >= 1:
                        rhs = (1 - y[box_1, y_step]) * self.data.boxes.iloc[box_1][
                            "width"
                        ]

                        cqm.add_constraint(
                            lhs - rhs <= 0,
                            penalty=self.penalty,
                            label=f"NonOverlappingY_WithRotation_{box_1}_{y_step}",
                        )

    def solve(self, solve_func, **config):
        self.optimize(solve_func=solve_func, **config)

    def optimize(self, solve_func, **config):

        lagrange_multiplier = config.get("lagrange_multiplier", 10)

        # Sample the QUBO
        qubo, inverter = FromCQM(self.model).to_matrix(lagrange_multiplier=lagrange_multiplier)
        start_time = time.time()
        sampleset = solve_func(qubo, **config)
        self.runtime = time.time() - start_time

        # Extract the best solution
        best_sample = sampleset.first.sample
        self.best_sample = inverter(best_sample)
        best_energy = sampleset.first.energy
        self.best_energy = best_energy

    def check_if_feasible(self):

        # check if box is placed at most on time
        OneBoxInTruck = self.check_constraint_at_most_one_box()

        # check if no box is overlapping
        NonOverlappingBoxes = self.check_constraints_overlapping()

        # check if there is at most one box per step
        OneBoxPerSpace = self.check_constraints_only_one_box_per_space()

        # Check if neccessary
        # BoxInsideTruck = self.check_constraints_box_inside_truck()

        violated_constraints = [*OneBoxInTruck, *NonOverlappingBoxes, *OneBoxPerSpace]

        # if len(violated_constraints) == 0:
        #     print("Found feasible solution with qubo")
        # else:
        #     print("NO feasible solution with qubo")
        #     print(violated_constraints)

        return violated_constraints
    
    def check_constraints_box_inside_truck(self):
        violated_constraints = []

        for i in self.boxes:
            rotation_key = f"r_{i}"
            rotation = 0
            if not rotation_key in self.fixed_varialbes:
                rotation = self.best_sample[rotation_key]

            if rotation < 0.5:
                x_length = self.data.boxes.iloc[i]["length"]
                y_length = self.data.boxes.iloc[i]["width"]
            else:
                x_length = self.data.boxes.iloc[i]["width"]
                y_length = self.data.boxes.iloc[i]["length"]

            for j in range(int(self.num_steps_x_by_box[i])):
                if self.best_sample[f"x_{i}_{j}"] >= 1:
                    if j < 0 or j + x_length > self.data.truck_length:
                        violated_constraints.append(("OutsideContainer_inX", i, j))

            # check violation in y
            for j in range(int(self.num_steps_y_by_box[i])):
                if self.best_sample[f"y_{i}_{j}"] >= 1:
                    if j < 0 or j + y_length > self.data.truck_width:
                        violated_constraints.append(("OutsideContainer_inY", i, j))

        return violated_constraints

    def check_constraint_at_most_one_box(self):
        violated_constraints = []

        for box in self.boxes:

            # check if box is assigned only once to segment
            box_applied_to_x = 0
            for x_step in range(self.num_steps_x_by_box[box]):
                box_applied_to_x += self.best_sample[f"x_{box}_{x_step}"]
            if box_applied_to_x > 1.5:
                violated_constraints.append(("BoxOnlyOnceInX", box))

            box_applied_to_y = 0
            for y_step in range(self.num_steps_y_by_box[box]):
                box_applied_to_y += self.best_sample[f"y_{box}_{y_step}"]
            if box_applied_to_y > 1.5:
                violated_constraints.append(("BoxOnlyOnceInY", box))

            # check if box fits inside truck
            diff_length_width = (
                self.data.boxes.iloc[box]["length"] - self.data.boxes.iloc[box]["width"]
            )

            # limit box to be inside truck in x axis
            max_u = self.num_steps_x_by_box[box]
            max_v = self.num_steps_y_by_box[box]
            box_assigned = 0
            rotation_key = f"r_{box}"
            rotation = 0
            if not rotation_key in self.fixed_varialbes:
                rotation = self.best_sample[rotation_key]

            if rotation >= 0.5:
                for u in range(max_u - diff_length_width, max_u):
                    if u > 0:
                        box_assigned += self.best_sample[f"x_{box}_{u}"]
                    else:
                        violated_constraints.append(("BoxNotInTruckLength", box))
                if box_assigned > 1:
                    violated_constraints.append(("BoxNotInTruckLength", box))
            else:
                for v in range(max_v - diff_length_width, max_v):
                    if v > 0:
                        box_assigned += self.best_sample[f"y_{box}_{v}"]
                    else:
                        violated_constraints.append(("BoxNotInTruckWidth", box))
                if box_assigned > 1:
                    violated_constraints.append(("BoxNotInTruckWidth", box))

        return violated_constraints

    def check_constraints_overlapping(self):
        violated_constraints = []
        for i in self.boxes:
            # determine how box is rotated or if rotation is fixed
            rotation_key = f"r_{i}"
            rotation = 0
            if not rotation_key in self.fixed_varialbes:
                rotation = self.best_sample[rotation_key]

            if rotation < 0.5:
                max_x_length = self.data.boxes.iloc[i]["length"]
                max_y_length = self.data.boxes.iloc[i]["width"]
            else:
                max_x_length = self.data.boxes.iloc[i]["width"]
                max_y_length = self.data.boxes.iloc[i]["length"]

            # check overlapping in x axis
            for j in range(int(self.num_steps_x_by_box[i])):
                if self.best_sample[f"x_{i}_{j}"] >= 1:

                    lhs = 0
                    for t in range(j, j + max_x_length):
                        for k in self.boxes:
                            if k != i:
                                if t < self.num_steps_x_by_box[k]:
                                    lhs += self.best_sample[f"x_{k}_{t}"]

                    if lhs > 0:
                        violated_constraints.append(("NonOverlapping_inX", i, j))

            # check violation in y
            for j in range(int(self.num_steps_y_by_box[i])):
                if self.best_sample[f"y_{i}_{j}"] >= 1:

                    lhs = 0

                    for k in self.boxes:
                        for t in range(
                            j, min(j + max_y_length, self.num_steps_y_by_box[k])
                        ):
                            if k != i:
                                lhs += self.best_sample[f"y_{k}_{t}"]

                    if lhs > 0:
                        violated_constraints.append(("NonOverlapping_inY", i, j))

        return violated_constraints

    def check_constraints_only_one_box_per_space(self):
        """
        This function checks if there is at most one box per step in each dimension.
        :return:
        """
        violated_constraints = []
        for j in range(self.num_x_variables):
            lhs = 0
            for i in self.boxes:
                if j < self.num_steps_x_by_box[i]:
                    lhs += self.best_sample[f"x_{i}_{j}"]
            if lhs > 1:
                violated_constraints.append(("OnlyOneBoxPerSpace_x", j))

        for j in range(self.num_y_variables):
            lhs = 0
            for i in self.boxes:
                if j < self.num_steps_y_by_box[i]:
                    lhs += self.best_sample[f"y_{i}_{j}"]
            if lhs > 1:
                violated_constraints.append(("OnlyOneBoxPerSpace_y", j))

        return violated_constraints

    def generate_solution_overview(self):
        """
        This function is used to standardize the model output and allow for plotting overview.
        :return:
        """

        runtime = self.runtime
        n_var = len(self.x.keys()) + len(self.y.keys()) + len(self.r.keys())
        n_constr = 0
        selected_boxes = {}

        for i in self.boxes:
            rotation_key = f"r_{i}"
            rotation = 0
            if not rotation_key in self.fixed_varialbes:
                rotation = self.best_sample[rotation_key]
            x_coord = -1
            y_coord = -1

            for j in range(int(self.num_steps_x_by_box[i])):
                x_var = f"x_{i}_{j}"
                if self.best_sample[x_var] == 1:
                    x_coord = j

            for j in range(int(self.num_steps_y_by_box[i])):
                y_var = f"y_{i}_{j}"
                if self.best_sample[y_var] == 1:
                    y_coord = j

                if x_coord > 0 and y_coord > 0:
                    selected_boxes[i] = {
                        "x_coord": x_coord,
                        "y_coord": y_coord,
                        "rotation": rotation,
                    }

        obj_val = 0
        for i in selected_boxes.keys():
            obj_val += self.data.boxes.iloc[i]["length"] * self.data.boxes.iloc[i]["width"]

        self.solution = Tl2D_Solution(
            model_name="Qubo",
            selected_boxes=selected_boxes,
            runtime=runtime,
            num_vars=n_var,
            num_constrs=n_constr,
            obj_val=obj_val,
            data=self.data,
            gap_to_optimal=None,
            violated_constraints=self.check_if_feasible(),
            energy=self.best_energy
        )
        # print(
        #     f"With qubo found {self.solution.objective_value} in {self.solution.runtime:.2f} seconds"
        # )
        return self.solution
