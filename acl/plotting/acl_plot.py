import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from acl.data.acl_data import ACLData
from typing import Set
import os
import re

class ACLPlot:
    """A plotting class which converts"""
    def __init__(self, evaluation, d_var_exists=True):
        self.data = evaluation.data
        self.solution = evaluation.original_solution
        print(self.solution)

        self.tot_pltfrm_length = None
        self.d_var_exists = d_var_exists

        """Define sub-figure sizes"""
        self.nmbr_pltfrms = [None for _ in range((len(self.data.Pl)+1) // 2)]
        for idx_lp, loading_pln in enumerate(self.data.Pl):
            if idx_lp % 2 == 0:
                tpl = (int(len(loading_pln)),)
            else:
                tpl += (int(len(loading_pln)),)
            self.nmbr_pltfrms[int(idx_lp // 2)] = tpl

        """Define axes for each subplot and the total plot in the end"""
        self.pltfrm_figures = []
        x_max_pltfrms_mem = 0
        for up_and_down_pltfrm in self.nmbr_pltfrms:
            x_max_pltfrms = max(up_and_down_pltfrm)

            x_max_pltfrms_mem += x_max_pltfrms
            fig, axes = plt.subplots(len(up_and_down_pltfrm), x_max_pltfrms, figsize=(12, 8))
            axes = np.atleast_2d(axes)
            self.pltfrm_figures.append((fig, axes))

        tot_fig, tot_axes = plt.subplots(len(self.nmbr_pltfrms[0]), x_max_pltfrms_mem, figsize=(12, 8))
        tot_axes = np.atleast_2d(tot_axes)
        self.pltfrm_figures.append((tot_fig, tot_axes))

        # Sets for checking that each platform has been considered while plotting
        self.pltfrm_check_set: Set[int] = set(self.data.P)
        self.checked_pltfrms: Set[int] = set()

        self.create_truck(self.solution)


    def parse_variables(self, string, variable_names):
        variable_pattern = '|'.join(variable_names)
        pattern = rf'({variable_pattern})_(\d+)(?:_(\d+))?(?:_(\d+))?'

        match = re.match(pattern, string)

        if match:
            variable_name = match.group(1) + "_variables"
            indices = tuple([int(group) for group in match.groups()[1:] if group is not None])
            if len(indices) == 1:
                indices = int(indices[0])
            return variable_name, indices
        else:
            raise ValueError("String does not match expected format")

    def convert_solution(self, solution):
        _variable_names = ["x",
                           "a",
                           "sp",
                           "d",
                           "d00",
                           "d01",
                           "d10",
                           "d11",
                           "ax00",
                           "ax01",
                           "ax10",
                           "ax11",
                           "xsp00",
                           "xsp01",
                           "xsp10",
                           "xsp11",
                           "axsp00",
                           "axsp01",
                           "axsp10",
                           "axsp11",
                           "axsp20",
                           "axsp21",
                           "axsp30",
                           "axsp31",
                           "axdd00",
                           "axdd01",
                           "axdd02",
                           "axdd03",
                           "axdd10",
                           "axdd11",
                           "axdd12",
                           "axdd13",
                           "axdd20",
                           "axdd21",
                           "axdd22",
                           "axdd23",
                           "axdd30",
                           "axdd31",
                           "axdd32",
                           "axdd33"
                           ]

        new_solution = {}
        for key in _variable_names:
            new_solution[key + "_variables"] = {}

        for (key_var, var_value) in solution.items():
            var_name, number_tuple = self.parse_variables(key_var, _variable_names)
            new_solution[var_name].update({number_tuple: var_value})

        if len(new_solution["d_variables"])==0:
            new_solution["d_variables"] = {i: 0 for i in range(len(self.data.P))}

        return new_solution


    def analyze_platform(self, platform, solution_dict):
        """If a car is assigned, calculate the weight, height and length texts based on the angled decision variable. In
         other words all relevant information for the specific platform and assigned car"""
        assigned_car = [
            k
            for k in range(self.data.K)
            if round(solution_dict["x_variables"][(k, platform)], 0) == 1
        ]
        if len(assigned_car) > 1:
            #stop if multiple cars are assigned to a platform
            exit(f"{len(assigned_car)} cars are assigned to one platform")
        elif len(assigned_car) == 0:
            return None

        #Define car parameters
        car_weight = self.data.vehicles[assigned_car[0]]["Weight"]
        car_height = self.data.vehicles[assigned_car[0]]["Height"] * 10**(-3) # transform mm to m
        car_length = self.data.vehicles[assigned_car[0]]["Length"] * 10**(-3) # transform mm to m
        car_model = self.data.vehicles[assigned_car[0]]["Modell"]

        allowed_weight = self.data.wp_max[platform]
        # check for angled variable
        if platform in self.data.Pa:
            angle_var = solution_dict["a_variables"][self.data.Pa.index(platform)]
        else:
            angle_var = 0
        # if the car is angled change allowed weight, new height and length
        if angle_var == 1:
            if self.d_var_exists:
                _lr_coefficient = 1 - sum(
                    [
                        self.data.lr_coefficient[self.data.vehicles[assigned_car[0]]["Class"]][0][0]
                        * solution_dict["d00_variables"][self.data.Pa.index(platform)],
                        self.data.lr_coefficient[self.data.vehicles[assigned_car[0]]["Class"]][0][1]
                        * solution_dict["d01_variables"][self.data.Pa.index(platform)],
                        self.data.lr_coefficient[self.data.vehicles[assigned_car[0]]["Class"]][1][0]
                        * solution_dict["d10_variables"][self.data.Pa.index(platform)],
                        self.data.lr_coefficient[self.data.vehicles[assigned_car[0]]["Class"]][1][1]
                        * solution_dict["d11_variables"][self.data.Pa.index(platform)],
                    ]
                )

                _hr_coefficient = 1 + sum(
                    [
                        self.data.hr_coefficient[self.data.vehicles[assigned_car[0]]["Class"]][0][0]
                        * solution_dict["d00_variables"][self.data.Pa.index(platform)],
                        self.data.hr_coefficient[self.data.vehicles[assigned_car[0]]["Class"]][0][1]
                        * solution_dict["d01_variables"][self.data.Pa.index(platform)],
                        self.data.hr_coefficient[self.data.vehicles[assigned_car[0]]["Class"]][1][0]
                        * solution_dict["d10_variables"][self.data.Pa.index(platform)],
                        self.data.hr_coefficient[self.data.vehicles[assigned_car[0]]["Class"]][1][1]
                        * solution_dict["d11_variables"][self.data.Pa.index(platform)],
                    ]
                )
            else:
                _lr_coefficient = 1 - self.data.lr_coefficient_no_d[self.data.vehicles[assigned_car[0]]["Class"]]

                _hr_coefficient = 1 + self.data.hr_coefficient_no_d[self.data.vehicles[assigned_car[0]]["Class"]]

            angle = 25
            allowed_weight = self.data.wa_max[self.data.Pa.index(platform)]
            weight_text = f"W_a={round(car_weight,0)}/{allowed_weight} kg |{round(car_weight * 100 / allowed_weight, 0)}%"

            car_height = car_height * _hr_coefficient
            height_text = f"H_a={round(car_height,2)} m"

            car_length = car_length * _lr_coefficient
            length_text = f"L_a={round(car_length,2)} m"

        else:
            angle = 0
            weight_text = f"W_p={round(car_weight,0)}/{allowed_weight} kg |{round(car_weight * 100 / allowed_weight, 0)}%"

            height_text = f"H_p={round(car_height,2)} m"

            length_text = f"L_p={round(car_length,2)} m"

        sp_idx = np.where(np.array(self.data.Psp) == int(platform))[0]
        if len(sp_idx) == 0:
            # car can not be combined
            combined = False
            combined_text = False
        else:
            for _sp_idx in sp_idx:
                if solution_dict["sp_variables"][_sp_idx] == 1:
                    combined = True
                    combined_text = f"{self.data.Psp[_sp_idx]}"
                    allowed_weight = self.data.wc_max[_sp_idx]
                    weight_text = f"W_sp={round(car_weight,0)}/{allowed_weight} kg |{round(car_weight * 100 / allowed_weight, 0)}%"
                    break
                else:
                    combined = False
                    combined_text = False
        # define text colour (exceeding or not)
        if car_weight > allowed_weight:
            weight_text_color = "red"
        else:
            weight_text_color = "green"

        # check which way the vehicle faces
        if solution_dict["d_variables"][platform] == 0 or not self.d_var_exists:
            forward_variable = False
        else:
            forward_variable = True

        #add platform length to total Pl length
        self.tot_pltfrm_length += car_length

        return_dict = (
            angle,
            weight_text,
            height_text,
            length_text,
            weight_text_color,
            forward_variable,
            combined,
            combined_text,
            car_model,
        )
        return return_dict

    def rotate_vertices(self, vertices, rotation_matrix, rotation_axis):
        # rotate set of vertices in space by multiplying their vector by a rotation matrix
        rotated_vertices = []
        for vertex in vertices:
            rotated_vertex = rotation_matrix.dot(np.array(vertex)) - np.array(
                rotation_axis
            )
            rotated_vertices.append(rotated_vertex)
        return rotated_vertices

    def plot_car(
        self,
        ax,
        ax_tot,
        weight_text,
        height_text,
        length_text,
        text_color,
        pltfrm_length_txt,
        pltfrm_height_txt,
        Length,
        Height,
        alpha,
        forward_facing,
        combined,
        combined_text,
        car_modell,
    ):

        if alpha == 0:
            rotation = False
        else:
            rotation = True

        """create tires"""
        num_sides = 32
        angle_step = 2 * np.pi / num_sides

        radius = 0.25 * Length/2
        min_y_point = -0.15
        middle_point_left = (-Length / 2 * 2 / 5, min_y_point)
        middle_point_right = (Length / 2 * 2 / 5, min_y_point)

        # Initialize list to store vertices
        hexadecagon_vertices_left = []
        hexadecagon_vertices_right = []

        # Calculate coordinates of each vertex
        for i in range(num_sides):
            x = radius * np.cos(i * angle_step) + middle_point_left[0]
            y = radius * np.sin(i * angle_step) + middle_point_left[1]
            hexadecagon_vertices_left.append((x, y))
        # Add the first vertex again to close the shape
        hexadecagon_vertices_left.append(hexadecagon_vertices_left[0])

        for i in range(num_sides):
            x = radius * np.cos(i * angle_step) + middle_point_right[0]
            y = radius * np.sin(i * angle_step) + middle_point_right[1]
            hexadecagon_vertices_right.append((x, y))
        # Add the first vertex again to close the shape
        hexadecagon_vertices_right.append(hexadecagon_vertices_right[0])

        """create body"""
        car_body_vertices = [(-Length / 2, min_y_point)]
        steps = [
            (0, Height * 0.5),
            (0.8 * Length / 5, 0),
            (0, -Height * 0.13),
            (-0.8 * Length / 5, 0),
            (0, Height * 0.13),
            (Length / 5, 0),
            (Length / 5, Height * 0.5),
            (0, -Height * 0.5),
            (-Length / 5, 0),
            (Length / 5, Height * 0.5),
            (Length * 3 / 10, 0),
            (Length / 10, -Height * 0.5),
            (-Length * 4 / 10, 0),
            (Length * 3 / 10, 0),
            (0, Height * 0.5),
            (Length / 10, -Height * 0.5),
            (Length / 5, 0),
            (0, -Height * 0.5),
        ]
        # for all etstps create total vector of each step plus all previouse
        for step in steps:
            car_body_vertices.append(
                (car_body_vertices[-1][0] + step[0], car_body_vertices[-1][1] + step[1])
            )
        car_body_vertices.append(car_body_vertices[0])

        # if not forward facoing, mirror the x axis
        # in addition the rotation angle must be taking with a minus to be consistent if the car si turned or not
        if not forward_facing:
            car_body_vertices = [(-x, y) for (x, y) in car_body_vertices]
        else:
            alpha = -alpha

        # Rotate if alpha!=0
        if rotation:
            # Define rotation matrix
            rotation_matrix = np.array(
                [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]
            )
            rotation_tire = hexadecagon_vertices_right
            rotation_axis = [i[1] for i in rotation_tire]
            rotation_axis = (
                0,
                hexadecagon_vertices_right[rotation_axis.index(min(rotation_axis))][1]
                / 3,
            )

            # Apply rotation transformation to car body vertices
            car_body_vertices = self.rotate_vertices(
                car_body_vertices, rotation_matrix, rotation_axis
            )

            # Apply rotation transformation to left tire vertices
            hexadecagon_vertices_left = self.rotate_vertices(
                hexadecagon_vertices_left, rotation_matrix, rotation_axis
            )

            # Apply rotation transformation to right tire vertices
            hexadecagon_vertices_right = self.rotate_vertices(
                hexadecagon_vertices_right, rotation_matrix, rotation_axis
            )

        # Apply car body vertices
        car_body_outline = Line2D(
            [vertex[0] for vertex in car_body_vertices],
            [vertex[1] for vertex in car_body_vertices],
            linewidth=2,
            color="black",
        )
        # Apply left tire vertices
        left_tire_outline = Line2D(
            [vertex[0] for vertex in hexadecagon_vertices_left],
            [vertex[1] for vertex in hexadecagon_vertices_left],
            linewidth=2,
            color="black",
        )
        # Apply left tire vertices
        right_tire_outline = Line2D(
            [vertex[0] for vertex in hexadecagon_vertices_right],
            [vertex[1] for vertex in hexadecagon_vertices_right],
            linewidth=2,
            color="black",
        )
        # Same for total truck plot
        car_body_outline_tot = Line2D(
            [vertex[0] for vertex in car_body_vertices],
            [vertex[1] for vertex in car_body_vertices],
            linewidth=2,
            color="black",
        )
        # Same for total truck plot
        left_tire_outline_tot = Line2D(
            [vertex[0] for vertex in hexadecagon_vertices_left],
            [vertex[1] for vertex in hexadecagon_vertices_left],
            linewidth=2,
            color="black",
        )
        # Same for total truck plot
        right_tire_outline_tot = Line2D(
            [vertex[0] for vertex in hexadecagon_vertices_right],
            [vertex[1] for vertex in hexadecagon_vertices_right],
            linewidth=2,
            color="black",
        )

        ax.add_line(car_body_outline)
        ax.add_line(left_tire_outline)
        ax.add_line(right_tire_outline)

        # This can be used if the car should be filled out (for some measure)
        """ax.fill_betweenx(
            [vertex[1] for vertex in car_body_vertices],
            [vertex[0] for vertex in car_body_vertices],
            where=[vertex[1] <= colour_cutoff for vertex in car_body_vertices],
            color="lightblue",
        )"""

        # text for cars that take up multiple platforms
        if combined:
            ax.text(
                0,
                0.15,
                "SP ->" + combined_text,
                fontsize=8,
                ha="center",
                va="center",
                color="black",
            )
        # weight/max weight | ()% (weight text)
        ax.text(
            -1.4, 1.08, weight_text, fontsize=8, ha="left", va="center", color=text_color
        )
        # height/max height | ()% (height text)
        ax.text(
            -1.4,
            0.93,
            height_text,
            fontsize=8,
            ha="left",
            va="center",
            color="black",
        )
        # legnth/max length | ()% (length text)
        ax.text(
            -1.4, 0.78, length_text, fontsize=8, ha="left", va="center", color="black"
        )
        if pltfrm_length_txt is not None:
            ax.text(
                -1.4,
                -0.56,
                pltfrm_length_txt[0],
                fontsize=8,
                ha="left",
                va="center",
                color=pltfrm_length_txt[1],
            )

        if pltfrm_height_txt is not None:
            ax.text(
                -1.4,
                -0.42,
                pltfrm_height_txt[0],
                fontsize=8,
                ha="left",
                va="center",
                color=pltfrm_height_txt[1],
            )
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1)
        #ax.set_title(f"Modell: {car_modell}")
        ax.set_aspect("equal")
        ax.axis("off")

        ax_tot.add_line(car_body_outline_tot)
        ax_tot.add_line(left_tire_outline_tot)
        ax_tot.add_line(right_tire_outline_tot)

        # the text for the total_trailer.png have been left out for a less cluttered image
        # but can reintroduced if desired
        """ax_tot.text(
            -1, 1, weight_text, fontsize=10, ha="center", va="center", color=text_color
        )
        ax_tot.text(
            -1.4, 0.8, height_text, fontsize=10, ha="center", va="center", color=text_color
        )

        ax_tot.text(
            -1.4, 0.6, length_text, fontsize=10, ha="center", va="center", color=text_color
        )

        if pltfrm_length_txt is not None:
            ax_tot.text(
                -1, -0.6, pltfrm_length_txt, fontsize=10, ha="center", va="center", color=text_color
            )
        if pltfrm_height_txt is not None:
            ax_tot.text(
                -1, -0.4, pltfrm_height_txt, fontsize=10, ha="center", va="center", color=text_color
            )"""
        ax_tot.set_xlim(-1.2, 1.2)
        ax_tot.set_ylim(-0.6, 1)
        ax_tot.set_title(f"Modell: {car_modell}")
        ax_tot.set_aspect("equal")
        ax_tot.axis("off")


    def calculate_car_height(self, _h_indices, _solution_dict):
        car_heights = 0
        for pltf in self.data.Ph[_h_indices]:
            # sum the total height of all cars assigned to the same height constraint as the pltfrm under inspection
            height_restricted_cars = [
                k
                for k in range(self.data.K)
                if _solution_dict["x_variables"][(k, pltf)] == 1
            ]
            if len(height_restricted_cars) != 0:
                # check if platform can be angled
                if pltf in self.data.Pa:
                    # check if platform is angled in the solution
                    if (
                            _solution_dict["a_variables"][
                                self.data.Pa.index(pltf)
                            ]
                            == 1
                    ):
                        if self.d_var_exists:
                            __hr_coefficient = 1 + sum(
                                [
                                    self.data.hr_coefficient[
                                        self.data.vehicles[height_restricted_cars[0]][
                                            "Class"
                                        ]
                                    ][0][0]
                                    * _solution_dict["d00_variables"][
                                        self.data.Pa.index(pltf)
                                    ],
                                    self.data.hr_coefficient[
                                        self.data.vehicles[height_restricted_cars[0]][
                                            "Class"
                                        ]
                                    ][0][1]
                                    * _solution_dict["d01_variables"][
                                        self.data.Pa.index(pltf)
                                    ],
                                    self.data.hr_coefficient[
                                        self.data.vehicles[height_restricted_cars[0]][
                                            "Class"
                                        ]
                                    ][1][0]
                                    * _solution_dict["d10_variables"][
                                        self.data.Pa.index(pltf)
                                    ],
                                    self.data.hr_coefficient[
                                        self.data.vehicles[height_restricted_cars[0]][
                                            "Class"
                                        ]
                                    ][1][1]
                                    * _solution_dict["d11_variables"][
                                        self.data.Pa.index(pltf)
                                    ],
                                ]
                            )
                        else:
                            _hr_coefficient = 1 + self.hr_coefficient[self.data.vehicles[height_restricted_cars[0]]["Class"]]

                        car_heights += (
                                self.data.vehicles[height_restricted_cars[0]]["Height"]
                                * __hr_coefficient
                        )
                    else:
                        car_heights += self.data.vehicles[height_restricted_cars[0]][
                            "Height"
                        ]
                else:
                    car_heights += self.data.vehicles[height_restricted_cars[0]][
                        "Height"
                    ]
            else:
                continue
        return car_heights

    def create_truck(self, solution_dict):
        solution_dict = self.convert_solution(solution_dict)

        # For each platform p in Pl plot the car on its position
        x_counter = 0
        for idx_lp, loading_pln in enumerate(self.data.Pl):
            (_fig, _axes) = self.pltfrm_figures[int(idx_lp // 2)]

            nmbr_up_down_plt = self.nmbr_pltfrms[int(idx_lp // 2)]

            #set total pltfrm length to zero at start of each Pl
            self.tot_pltfrm_length = 0

            for fig_idx, platform in enumerate(loading_pln):
                # Dicard the platform for the check
                self.pltfrm_check_set.discard(platform)
                self.checked_pltfrms.add(platform)

                # Define axes indices of total truck plot
                tot_ax_idx = idx_lp % 2
                tot_ax_idy = fig_idx + x_counter

                # Define axes indices of truck plot
                if idx_lp % 2 == 1:
                    ax_idx = 1
                    ax_idy = fig_idx
                else:
                    ax_idx = 0
                    ax_idy = fig_idx

                # get the variable values of the car assigned
                var_values = self.analyze_platform(
                    platform=platform, solution_dict=solution_dict
                )

                if var_values is not None:
                    (
                        angle,
                        weight_text,
                        height_text,
                        length_text,
                        text_color,
                        forward_variable,
                        combined,
                        combined_text,
                        car_model,
                    ) = var_values

                # if no car is assigned still print H_p (and, or) L_p if the appropriate platform is considered, but do
                # not plot a car if var_values=None thereby no car assigned
                elif var_values is None and fig_idx == len(loading_pln) - 1 and idx_lp % 2 == 0:
                    _axes[ax_idx, ax_idy].text(
                        -0.08,
                        0.02,
                        f"L={round(self.tot_pltfrm_length, 2)}/{self.data.L_max[idx_lp]*10**(-3)} m |{round(self.tot_pltfrm_length * 100 / (self.data.L_max[idx_lp] *10**(-3)), 2)}%",
                        fontsize=8,
                        ha="left",
                        va="center",
                        color="black",
                    )

                    h_indices = np.where(np.array([platform in sublist for sublist in self.data.Ph]))[0][0]

                    tot_height = self.calculate_car_height(_h_indices=h_indices,
                                                           _solution_dict=solution_dict)
                    _axes[ax_idx, ax_idy].text(
                        -0.08,
                        0.11,
                        f"H={round(tot_height * 10**(-3), 2)}/{self.data.H_max[fig_idx]*10**(-3)} m |{round(tot_height * 100 / self.data.H_max[fig_idx], 2)}%",
                        fontsize=8,
                        ha="left",
                        va="center",
                        color="black",
                    )
                    _axes[ax_idx, ax_idy].axis("off")
                    _tot_axis = self.pltfrm_figures[-1][1]
                    _tot_axis[tot_ax_idx, tot_ax_idy].axis("off")
                    continue

                elif var_values is None and fig_idx == len(loading_pln) - 1:
                    _axes[ax_idx, ax_idy].text(
                        -0.08,
                        0.02,
                        f"L={round(self.tot_pltfrm_length, 2)}/{self.data.L_max[idx_lp]*10**(-3)} m |{round(self.tot_pltfrm_length * 100 / (self.data.L_max[idx_lp] *10**(-3)), 2)}%",
                        fontsize=8,
                        ha="left",
                        va="center",
                        color="black",
                    )

                    _axes[ax_idx, ax_idy].axis("off")
                    _tot_axis = self.pltfrm_figures[-1][1]
                    _tot_axis[tot_ax_idx, tot_ax_idy].axis("off")

                    continue

                elif var_values is None and idx_lp % 2 == 0:

                    h_indices = np.where(np.array([platform in sublist for sublist in self.data.Ph]))[0][0]

                    tot_height = self.calculate_car_height(_h_indices=h_indices,
                                                           _solution_dict=solution_dict)
                    _axes[ax_idx, ax_idy].text(
                        -0.08,
                        0.11,
                        f"H={round(tot_height *10**(-3) , 2)}/{self.data.H_max[fig_idx]*10**(-3)} |{round(tot_height * 100 / self.data.H_max[fig_idx], 2)}%",
                        fontsize=8,
                        ha="left",
                        va="center",
                        color="black",
                    )
                    _axes[ax_idx, ax_idy].axis("off")
                    _tot_axis = self.pltfrm_figures[-1][1]
                    _tot_axis[tot_ax_idx, tot_ax_idy].axis("off")
                    continue

                else:
                    _axes[ax_idx, ax_idy].axis("off")
                    _tot_axis = self.pltfrm_figures[-1][1]
                    _tot_axis[tot_ax_idx, tot_ax_idy].axis("off")
                    continue

                # if the last platform of the Pl and the top floor of the truck is considered, the Pl length and Ph
                # height constraints are printed
                if fig_idx == len(loading_pln) - 1 and idx_lp % 2 == 0:
                    tot_length = self.tot_pltfrm_length
                    if tot_length > self.data.L_max[idx_lp]:
                        _pltfrm_length_colour = "red"
                    else:
                        _pltfrm_length_colour = "green"
                    _pltfrm_length_txt = f"L={round(tot_length,2)}/{self.data.L_max[idx_lp] *10**(-3)} m |{round(tot_length * 100 / (self.data.L_max[idx_lp] *10**(-3)), 2)}%"

                    #calculate total height for Ph
                    h_indices = np.where(np.array([platform in sublist for sublist in self.data.Ph]))[0][0]

                    tot_height = self.calculate_car_height(_h_indices=h_indices,
                                                           _solution_dict=solution_dict)
                    # check if total height is above maximum allowed height
                    if tot_height > self.data.L_max[idx_lp]:
                        _pltfrm_height_colour = "red"
                    else:
                        _pltfrm_height_colour = "green"

                    _pltfrm_height_txt = f"H={round(tot_height *10**(-3),2)}/{self.data.H_max[fig_idx] *10**(-3)} m |{round(tot_height * 100 / self.data.H_max[fig_idx], 2)}%"

                    self.plot_car(
                        ax=_axes[ax_idx, ax_idy],
                        ax_tot=self.pltfrm_figures[-1][1][tot_ax_idx, tot_ax_idy],
                        weight_text=weight_text,
                        height_text=height_text,
                        length_text=length_text,
                        text_color=text_color,
                        pltfrm_length_txt=(_pltfrm_length_txt, _pltfrm_length_colour),
                        pltfrm_height_txt=(_pltfrm_height_txt, _pltfrm_height_colour),
                        Length=2*0.76,
                        Height=1*0.76,
                        alpha=2 * np.pi / 360 * angle,
                        forward_facing=forward_variable,
                        combined=combined,
                        combined_text=combined_text,
                        car_modell=car_model,
                    )
                elif fig_idx == len(loading_pln) - 1:
                    tot_length = self.tot_pltfrm_length
                    if tot_length > self.data.L_max[idx_lp]:
                        _pltfrm_length_colour = "red"
                    else:
                        _pltfrm_length_colour = "green"
                    _pltfrm_length_txt = f"L={round(tot_length,2)}/{self.data.L_max[idx_lp]*10**(-3)} m |{round(tot_length * 100 / (self.data.L_max[idx_lp] *10**(-3)), 2)}%"

                    self.plot_car(
                        ax=_axes[ax_idx, ax_idy],
                        ax_tot=self.pltfrm_figures[-1][1][tot_ax_idx, tot_ax_idy],
                        weight_text=weight_text,
                        height_text=height_text,
                        length_text=length_text,
                        text_color=text_color,
                        pltfrm_length_txt=(_pltfrm_length_txt, _pltfrm_length_colour),
                        pltfrm_height_txt=None,
                        Length=2*0.76,
                        Height=1*0.76,
                        alpha=2 * np.pi / 360 * angle,
                        forward_facing=forward_variable,
                        combined=combined,
                        combined_text=combined_text,
                        car_modell=car_model,
                    )
                elif idx_lp % 2 == 0:
                    h_indices = np.where(np.array([platform in sublist for sublist in self.data.Ph]))[0][0]

                    tot_height = self.calculate_car_height(_h_indices=h_indices,
                                                           _solution_dict=solution_dict)
                    if tot_height > self.data.H_max[fig_idx]:
                        _pltfrm_height_colour = "red"
                    else:
                        _pltfrm_height_colour = "green"
                    _pltfrm_height_txt = f"H={round(tot_height *10**(-3),2)}/{self.data.H_max[fig_idx]*10**(-3)} m |{round(tot_height * 100 / self.data.H_max[fig_idx], 2)}%"

                    self.plot_car(
                        ax=_axes[ax_idx, ax_idy],
                        ax_tot=self.pltfrm_figures[-1][1][tot_ax_idx, tot_ax_idy],
                        weight_text=weight_text,
                        height_text=height_text,
                        length_text=length_text,
                        text_color=text_color,
                        pltfrm_length_txt=None,
                        pltfrm_height_txt=(_pltfrm_height_txt, _pltfrm_height_colour),
                        Length=2*0.76,
                        Height=1*0.76,
                        alpha=2 * np.pi / 360 * angle,
                        forward_facing=forward_variable,
                        combined=combined,
                        combined_text=combined_text,
                        car_modell=car_model,
                    )
                else:
                    self.plot_car(
                        ax=_axes[ax_idx, ax_idy],
                        ax_tot=self.pltfrm_figures[-1][1][tot_ax_idx, tot_ax_idy],
                        weight_text=weight_text,
                        height_text=height_text,
                        length_text=length_text,
                        text_color=text_color,
                        pltfrm_length_txt=None,
                        pltfrm_height_txt=None,
                        Length=2*0.76,
                        Height=1*0.76,
                        alpha=2 * np.pi / 360 * angle,
                        forward_facing=forward_variable,
                        combined=combined,
                        combined_text=combined_text,
                        car_modell=car_model,
                    )
            if idx_lp % 2 == 1:
                # update x_counter variable for total truck plot
                x_counter += max(nmbr_up_down_plt)

    def check_pltfrm_set(self):
        if not self.pltfrm_check_set:
            print("pltfrm_check_set is empty")
        # Check if checked_pltfrms is identical to set(data.P)
        if self.checked_pltfrms == set(self.data.P):
            print("checked_pltfrms is identical to set(data.P)")
        else:
            print("checked_pltfrms is not identical to set(data.P)")

    def save_fig(
        self,
        name,
    ):
        """Save plots of individual trailers and total truck to the file plots"""

        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")

        trck_lst = ""
        for trck in self.data.truck_list:
            trck_lst += trck
        filepath += f"/K={self.data.K}Seed={self.data.seed}TrckLst={trck_lst}" + f"{name}"
        os.mkdir(filepath)

        for idx, (figg, axees) in enumerate(self.pltfrm_figures):
            png_name = f"trailer_{idx}.png"
            if idx == len(self.pltfrm_figures) - 1:
                png_name = "total_trailer.png"
            figg.savefig(
                os.path.join(
                    filepath,
                    png_name,
                )
            )
            plt.close(figg)

    def show_single_version(self, version=1):
        version_idx = version - 1  # Adjust for zero-based indexing
        if version_idx < 0 or version_idx >= len(self.pltfrm_figures):
            print(f"Invalid version number: {version}")
            return
        # Get the specified figure and axes
        figg, axees = self.pltfrm_figures[version_idx]
        # Set the suptitle for the figure
        if version_idx == len(self.pltfrm_figures) - 1:
            name = ""
            for _name_idx in range(len(self.pltfrm_figures) - 1):
                if _name_idx == len(self.pltfrm_figures) - 2:
                    name += self.data.truck_list[_name_idx] + ""
                else:
                    name += self.data.truck_list[_name_idx] + " and "
        else:
            name = self.data.truck_list[version_idx]
        figg.suptitle(name + f" car seed = {self.data.seed}")
        # Close all other figures
        for idx, (f, a) in enumerate(self.pltfrm_figures):
            if idx != version_idx:
                plt.close(f)
        plt.show()

    def show(self, version=1):
        if version == 1:
            versions = [i for i in range(len(self.data.truck_list))]
        else:
            print("version works only for 1 truck")
            versions = [i + len(self.data.truck_list) for i in range(len(self.data.truck_list))]
        for version in versions:
            # Get the specified figure and axes
            figg, axees = self.pltfrm_figures[version]
            # Set the suptitle for the figure
            if version == len(self.pltfrm_figures) - 1:
                name = ""
                for _name_idx in range(len(self.pltfrm_figures) - 1):
                    if _name_idx == len(self.pltfrm_figures) - 2:
                        name += self.data.truck_list[_name_idx] + ""
                    else:
                        name += self.data.truck_list[_name_idx] + " and "
            else:
                name = self.data.truck_list[version]
            figg.suptitle(name + f" car seed = {self.data.seed}")
            # Close all other figures
        for idx, (f, a) in enumerate(self.pltfrm_figures):
            if idx not in versions:
                plt.close(f)
        plt.show()



def single_plot(complexity, fit_complexity, x_data, fit_x_data):
    for name, _compl in complexity.items():
        plt.scatter(x_data, _compl, label=name)

    for name, _compl in fit_complexity.items():
        plt.plot(fit_x_data, _compl)

    plt.xlabel("Number of cars k")
    plt.ylabel("Number of Variables")
    plt.legend()
    plt.title(f"Linear Optimised Qubo Complexity")
    plt.show()

def subplot_plots(complexity, fit_complexity, x_data, fit_x_data, ax, Qubo, Old_slack, P_linear):
    for name, _compl in complexity.items():
        ax.scatter(x_data, _compl, label=name)

    for name, _compl in fit_complexity.items():
        ax.plot(fit_x_data, _compl)

    ax.set_xlabel("Number of cars k")
    ax.set_ylabel("Number of Variables")
    ax.legend()
    ax.set_title(f"qubo={Qubo}, old slack={Old_slack}, p_linear={P_linear}, p=k_var/2")