from acl.data.acl_data import ACLData
from typing import TypedDict, Union

import numpy as np
import itertools
import re
from abstract.evaluation.abstract_evaluation import AbstractEvaluation


# have to change to my format
SolutionType = dict[
    str,
    Union[dict[tuple[int, int, int], int], dict[tuple[int, int], int], dict[int, int]],
]


class ACLEvaluation(AbstractEvaluation):
    """
    Evaluation of a solution to the PAS problem.
    """

    def __init__(
        self,
        data: ACLData,
        solution: SolutionType,
        convert_solution: bool = True,
    ):
        self.data = data
        self.original_solution = solution

        if convert_solution:
            self.solution = self.set_solution(solution)
        else:
            self.solution = solution

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

    def set_solution(self, solution):
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
            new_solution[key+"_variables"] = {}

        for (key_var, var_value) in solution.items():
            var_name, number_tuple = self.parse_variables(key_var, _variable_names)
            new_solution[var_name].update({number_tuple: var_value})

        if len(new_solution["d_variables"])==0:
            new_solution["d_variables"] = {i: 0 for i in range(len(self.data.P))}

        return new_solution

    def check_solution(self):
        error: dict[str, list[str]] = {
            "One car per platform": [],
            "One platform per car": [],
            "Max one combination per pltfrm": [],
            "Multiple cars on combined platform": [],
            "Maximal loading platform length": [],
            "Anglement in combined platforms": [],
            "Height constraint of platforms": [],
            "Non-combinable+non-angleable platform weight limit": [],
            "Combinable+non-angleable platform weight limit": [],
            "Non-combinable+angleable platform weight limit": [],
            "Combinable+angleable platform weight limit": [],
            "Loading platform weight limit": [],
            "Truck/trailer weight limit": [],
            "Total vehicle weight limit": [],
        }
        error = self.check_assignment_c1(error)
        error = self.check_assignment_c2(error)
        error = self.check_assignment_c3(error)
        error = self.check_assignment_c4(error)
        error = self.check_length_constraints(error)
        error = self.check_height_constraint(error)
        error = self.check_weight_constraints(error)

        return error

    def get_objective(self):
        """
        Define objective function.
        Returns: the objective value if build=False and sets the maximisation objective for self.__moedel if build=True
        """
        x_variables = self.solution["x_variables"]

        obj_func = -sum(
            [
                x_variables[(k, p)]
                for (k, p) in itertools.product(
                    range(self.data.K), range(len(self.data.P))
                )
            ]
        )

        return obj_func

    def check_assignment_c1(self, error):
        """
        Checks the assignment constraints:
        One car per platform
        """
        x_variables = self.solution["x_variables"]
        for p in self.data.P:
            constr = sum(x_variables[(k, p)] for k in range(self.data.K))
            if constr > 1:
                error["One car per platform"].append(
                    f"Platform {p} has too many cars ({constr}) assigned to it"
                )

        return error

    def check_assignment_c2(self, error):
        """
        Checks the assignment constraints:
        One platform per car
        """
        x_variables = self.solution["x_variables"]
        for k in range(self.data.K):
            constr = sum(x_variables[(k, p)] for p in self.data.P)
            if constr > 1:
                error["One platform per car"].append(
                    f"Car {k} has too many platforms ({constr}) assigned to it"
                )

        return error

    def check_assignment_c3(self, error):
        """
        Checks the assignment constraints:
        Max one combination possible per platform (added by constantin) and
        """
        sp_variables = self.solution["sp_variables"]
        for p in self.data.P:
            q_indices = np.where(np.array(self.data.Psp) == int(p))[0]
            constr = sum(sp_variables[_h] for _h in q_indices)
            if constr > 1:
                error["Max one combination per pltfrm"].append(
                    f"Platform {p} has been combined with ({constr}) platforms instead of only 1"
                )

        return error

    def check_assignment_c4(self, error):
        """
        Checks the assignment constraints:
        Only one car assigned to all platforms if combined
        """
        x_variables = self.solution["x_variables"]
        sp_variables = self.solution["sp_variables"]

        for q_idx, q in enumerate(self.data.Psp):
            constr = sum(
                [
                    x_variables[(k, p)]
                    for (k, p) in itertools.product(range(self.data.K), q)
                ]
            )
            max_constr = len(q) * (1 - sp_variables[q_idx]) + sp_variables[q_idx]
            if constr > max_constr:
                error["Multiple cars on combined platform"].append(
                    f"{constr} cars have been assigned to the q_idx={q_idx} combined platform"
                )

        return error

    def check_length_constraints(self, error):
        """
        Length constraints

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """

        x_variables = self.solution["x_variables"]
        a_variables = self.solution["a_variables"]
        sp_variables = self.solution["sp_variables"]
        d_variables = self.solution["d_variables"]

        """The total length of the cars on a platform can not be larger than the maximally allowed Length"""
        # constraint 5
        for L_idx, L in enumerate(self.data.Pl):
            L_set = set([i for i in L])

            angled_L = sum(
                [
                    x_variables[(k, p)] * self.data.vehicles[k]["Length"]
                    - self.data.vehicles[k]["Length"]
                    * x_variables[(k, p)]
                    * a_variables[self.data.Pa.index(p)]
                    * (
                        self.data.lr_coefficient[self.data.vehicles[k]["Class"]][0][0]
                        * (1 - d_variables[p])
                        * (1 - d_variables[self.data.Pa_nu[self.data.Pa.index(p)]])
                        + self.data.lr_coefficient[self.data.vehicles[k]["Class"]][0][1]
                        * (1 - d_variables[p])
                        * d_variables[self.data.Pa_nu[self.data.Pa.index(p)]]
                        + self.data.lr_coefficient[self.data.vehicles[k]["Class"]][1][0]
                        * d_variables[p]
                        * (1 - d_variables[self.data.Pa_nu[self.data.Pa.index(p)]])
                        + self.data.lr_coefficient[self.data.vehicles[k]["Class"]][1][1]
                        * d_variables[p]
                        * d_variables[self.data.Pa_nu[self.data.Pa.index(p)]]
                    )
                    for (k, p) in itertools.product(
                        range(self.data.K), list(L_set.intersection(self.data.Pa_set))
                    )
                ]
            )
            not_angled_L = sum(
                [
                    x_variables[(k, p)] * self.data.vehicles[k]["Length"]
                    for (k, p) in itertools.product(
                        range(self.data.K), list(L_set.difference(self.data.Pa_set))
                    )
                ]
            )
            constr = angled_L + not_angled_L
            max_constr = self.data.L_max[L_idx]

            if constr > max_constr:
                error["Maximal loading platform length"].append(
                    f"The total length {constr} of cars assigned to the loading platform L_idx={L_idx} larger that the maximally allowed {max_constr}"
                )

        """If platforms are combined to a larger platform, none of the individual platforms can be angled"""
        # constraint 6
        for q_idx, q_ in enumerate(self.data.Psp):
            q_set = set(q_)
            constr = sum(
                a_variables[self.data.Pa.index(p)]
                for p in q_set.intersection(self.data.Pa_set)
            )
            max_constr = len(q_) * (1 - sp_variables[q_idx])

            if constr > max_constr:
                error["Anglement in combined platforms"].append(
                    f"{constr} platforms are angled although they are defined to be used as a combined platform q_idx={q_idx}, where max_constr={max_constr} are allowed"
                )

        return error

    def check_height_constraint(self, error):
        """
        Height constraints

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        x_variables = self.solution["x_variables"]
        a_variables = self.solution["a_variables"]
        d_variables = self.solution["d_variables"]

        """The total height of the cars assigned to the platform sets of the set Ph, have to be smaller than H_max"""
        # constraint 7
        for H_idx, H in enumerate(self.data.Ph):
            H_set = set(H)

            non_angled_height = sum(
                x_variables[(k, p)] * self.data.vehicles[k]["Height"]
                for (k, p) in itertools.product(
                    range(self.data.K), list(H_set.difference(self.data.Pa_set))
                )
            )

            angled_height = sum(
                [
                    x_variables[(k, p)] * self.data.vehicles[k]["Height"]
                    + self.data.vehicles[k]["Height"]
                    * x_variables[(k, p)]
                    * a_variables[self.data.Pa.index(p)]
                    * (
                        self.data.hr_coefficient[self.data.vehicles[k]["Class"]][0][0]
                        * (1 - d_variables[p])
                        * (1 - d_variables[self.data.Pa_nu[self.data.Pa.index(p)]])
                        + self.data.hr_coefficient[self.data.vehicles[k]["Class"]][0][1]
                        * (1 - d_variables[p])
                        * d_variables[self.data.Pa_nu[self.data.Pa.index(p)]]
                        + self.data.hr_coefficient[self.data.vehicles[k]["Class"]][1][0]
                        * d_variables[p]
                        * (1 - d_variables[self.data.Pa_nu[self.data.Pa.index(p)]])
                        + self.data.hr_coefficient[self.data.vehicles[k]["Class"]][1][1]
                        * d_variables[p]
                        * d_variables[self.data.Pa_nu[self.data.Pa.index(p)]]
                    )
                    for (k, p) in itertools.product(
                        range(self.data.K), list(H_set.intersection(self.data.Pa_set))
                    )
                ]
            )
            constr = angled_height + non_angled_height
            max_constr = self.data.H_max[H_idx]

            if constr > max_constr:
                error["Height constraint of platforms"].append(
                    f"The height of the height contraint H_idx={H_idx} is {constr} which is larger than the maximal allowed height {max_constr}"
                )

        return error

    def check_weight_constraints(self, error):
        """
        Weight constraints

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        x_variables = self.solution["x_variables"]
        sp_variables = self.solution["sp_variables"]
        a_variables = self.solution["a_variables"]

        """Platforms than can not be angled, have to be checkd so that the weight
        of their assigned car does not exceed (gamma:) the weight limit of either
        the platform or the weight limit of a combined configuration with another
        platform"""
        for p in set(self.data.P).difference(self.data.Pa_set):
            # calculate the weight of the cars assigned to the platform p
            q_indices = np.where(np.array(self.data.Psp) == int(p))[0]

            weight = sum(
                [
                    self.data.vehicles[k]["Weight"] * x_variables[(k, p)]
                    for k in range(self.data.K)
                ]
            )

            # constraint 8
            # if q set is empty the total weight must be less than the wp_max of the platform
            if len(q_indices) == 0:
                max_pltfrm_weight = self.data.wp_max[p]
                if weight > max_pltfrm_weight:
                    error[f"Non-combinable+non-angleable platform weight limit"].append(
                        f"The weight {weight} of the non-combinable and non-angleable platform {p} exceeds its maximal weight limit wp_max={max_pltfrm_weight}"
                    )

            # constraint 9
            for q_idx in q_indices:
                gamma_constraint = sum(
                    [
                        sp_variables[q_idx]
                        * x_variables[(k, p)]
                        * self.data.wc_max[q_idx]
                        + (1 - sp_variables[q_idx])
                        * x_variables[(k, p)]
                        * self.data.wp_max[p]
                        for k in range(self.data.K)
                    ]
                )
                if weight > gamma_constraint:
                    error[f"Combinable+non-angleable platform weight limit"].append(
                        f"The weight {weight} of the combinable and non-angleable platform {p} exceeds its maximal weight limit gamma={gamma_constraint}"
                    )

        """Platforms that can be angled, are required to have a smaller weight than 
        the maximally allowed weight in an angled position. While at the same time, 
        if the platform is not angled the gamma factor checks the maximally allowed 
        load based on whether the platform can/has been combined with another to a 
        larger platform"""
        for p in self.data.Pa_set:
            q_indices = np.where(np.array(self.data.Psp) == int(p))[0]

            weight = sum(
                self.data.vehicles[k]["Weight"] * x_variables[(k, p)]
                for k in range(self.data.K)
            )

            # constraint 10
            if len(q_indices) == 0:
                max_angle_weight = (
                    a_variables[self.data.Pa.index(p)]
                    * self.data.wa_max[self.data.Pa.index(p)]
                    + (1 - a_variables[self.data.Pa.index(p)]) * self.data.wp_max[p]
                )
                if weight > max_angle_weight:
                    error[f"Non-combinable+angleable platform weight limit"].append(
                        f"The weight {weight} of the non-combinable and angleable platform {p} exceeds its maximal weight limit max_angle_weight={max_angle_weight}"
                    )

            # constraint 11
            for q_idx in q_indices:
                gamma_constraint = sum(
                    [
                        sp_variables[q_idx]
                        * x_variables[(k, p)]
                        * self.data.wc_max[q_idx]
                        + (1 - sp_variables[q_idx])
                        * x_variables[(k, p)]
                        * self.data.wp_max[p]
                        for k in range(self.data.K)
                    ]
                )
                angle_const = (
                    a_variables[self.data.Pa.index(p)]
                    * self.data.wa_max[self.data.Pa.index(p)]
                )

                anlge_gamma_constr = sum(
                    [
                        x_variables[(k, p)]
                        * a_variables[self.data.Pa.index(p)]
                        * (
                            sp_variables[q_idx] * self.data.wc_max[q_idx]
                            + (1 - sp_variables[q_idx]) * self.data.wp_max[p]
                        )
                        for k in range(self.data.K)
                    ]
                )
                max_sp_a_weight = -anlge_gamma_constr + angle_const + gamma_constraint
                if weight > max_sp_a_weight:
                    error[f"Combinable+angleable platform weight limit"].append(
                        f"The weight {weight} of the combinable and angleable platform {p} exceeds its maximal weight limit max_sp_a_weight={max_sp_a_weight}"
                    )

        """The weight of the cars on each platform L, must be smaller than the maximally allowed weight on L"""
        # constraint 12
        for L_idx, L in enumerate(self.data.Pl):
            platform_weight = sum(
                [
                    self.data.vehicles[k]["Weight"] * x_variables[(k, p)]
                    for (k, p) in itertools.product(range(self.data.K), L)
                ]
            )
            max_pl_weight = self.data.wl_max[L_idx]

            if platform_weight > max_pl_weight:
                error[f"Loading platform weight limit"].append(
                    f"The total weight {weight} on the loading platform {L_idx} exceeds its maximal weight limit max_pl_weight={max_pl_weight}"
                )

        """The total weight that is exerted on the truck or its trailers, must be smaller than their maximally allowed load"""
        # constraint 13
        for T_idx, T in enumerate(self.data.Pt):
            vehicle_weight = sum(
                [
                    self.data.vehicles[k]["Weight"] * x_variables[(k, p)]
                    for (k, p) in itertools.product(range(self.data.K), T)
                ]
            )
            max_t_weight = self.data.wt_max[T_idx]

            if vehicle_weight > max_t_weight:
                error[f"Truck/trailer weight limit"].append(
                    f"The total weight {weight} on the truck/trailer {T_idx} exceeds its maximal weight limit max_t_weight={max_t_weight}"
                )

        """The total weight of the vehicles on all platforms must be smaller than W_max"""
        # constraint 14
        total_weight = sum(
            self.data.vehicles[k]["Weight"] * x_variables[(k, p)]
            for (k, p) in itertools.product(range(self.data.K), self.data.P)
        )
        max_T_weight = self.data.W_max

        if total_weight > max_T_weight:
            error[f"Total vehicle weight limit"].append(
                f"The total weight {weight} on the vehicle exceeds its maximal weight limit max_T_weight={max_T_weight}"
            )

        return error
