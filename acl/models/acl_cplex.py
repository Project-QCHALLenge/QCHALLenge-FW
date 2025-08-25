import time
import itertools
import numpy as np

from docplex.mp.model import Model as CplexModel
from enum import Enum

from acl.data.acl_data import ACLData

class Sense(Enum):
    Min = "min"
    Max = "max"


class CplexACL(CplexModel):
    """
    Class to create a Cplex model, with maximally quadratic temrs, for the Autocarrier Loading Problem
    """

    def __init__(self, data: ACLData):
        """
        Initialize the Quad_CplexACL class with ACLData.

        Parameters:
        - data (ACLData): Input data for the model. data includes, truck and vehicle parameters.
        """
        self.data: ACLData = data

        # define decision parameter indices and initialise the cplex model
        self.__allowed_variables = self.allowed_variables()
        self.__model = CplexModel(name="ACL_quad")

        # define all decision variables of the model
        self.__x_variables = self.__model.binary_var_dict(
            keys=self.__allowed_variables["x"], name="x"
        )
        self.__a_variables = self.__model.binary_var_dict(
            keys=self.__allowed_variables["a"], name="a"
        )
        self.__sp_variables = self.__model.binary_var_dict(
            keys=self.__allowed_variables["sp"], name="sp"
        )
        self.__d_variables = self.__model.binary_var_dict(
            keys=self.__allowed_variables["d"], name="d"
        )

        #collect all variables in a dictionary
        self.__variables = {
            "x_variables": self.__x_variables,
            "a_variables": self.__a_variables,
            "sp_variables": self.__sp_variables,
            "d_variables": self.__d_variables,
        }

    def build_model(self):
        """
        Build the model
        """
        #objective function
        self.objective_function(variable=self.__x_variables)

        #assignement constraints
        self.assignment_constraints(
            x_variables=self.__x_variables, sp_variables=self.__sp_variables
        )
        #length constraint
        self.length_constraints(
            x_variables=self.__x_variables,
            a_variables=self.__a_variables,
            sp_variables=self.__sp_variables,
            d_variables=self.__d_variables,
        )
        #height constraint
        self.height_constraint(
            x_variables=self.__x_variables,
            a_variables=self.__a_variables,
            d_variables=self.__d_variables,

        )
        #weight constraint
        self.weight_constraints(
            x_variables=self.__x_variables,
            sp_variables=self.__sp_variables,
            a_variables=self.__a_variables,
        )

    def solve(self, **config):
        """
        Solve function, which firstly solves the optimisation problem and returns the solution.

        Returns: tuple(variable names), dict(variable names: dict(solution of variable))
        """
        TimeLimit = config.get("TimeLimit", 1)
        self.model.set_time_limit(TimeLimit)

        start_time = time.time()
        self.__model.solve()
        runtime = time.time() - start_time

        solution = {var.name: var.solution_value for var in self.__model.iter_variables()}

        return {"solution": solution, "runtime": runtime}

    def objective_function(self, variable, build=True):
        """
        Define objective function.
        Returns: the objective value if build=False and sets the maximisation objective for self.__moedel if build=True
        """
        obj_func = -sum(
            [
                variable[(k, p)]
                for (k, p) in itertools.product(
                    range(self.data.K), range(len(self.data.P))
                )
            ]
        )
        if build:
            self.__model.set_objective(sense=Sense.Max.value, expr=obj_func)
        else:
            return obj_func

    def assignment_constraints(self, x_variables, sp_variables, build=True):
        """
        Assignment Constraints

        Creates the assignment constraints for:
        -one car per platform,
        -one platform per car,
        -one combination possible per platform (added by me) and
        -only one car assigned to all platforms if combined

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        constr_dict = {}
        # constraint 1
        """Each platform have maximally one cars assigned to it"""
        for p in self.data.P:
            constr = sum(x_variables[(k, p)] for k in range(self.data.K))
            if build:
                self.__model.add_constraint(constr <= 1)
            constr_dict[f"const.1 p={p}"] = (constr, 1)

        # constraint 2
        """Each car can maximally be assigned to one platform"""
        for k in range(self.data.K):
            constr = sum(x_variables[(k, p)] for p in self.data.P)
            if build:
                self.__model.add_constraint(constr <= 1)
            constr_dict[f"const.2 k={k}"] = (constr, 1)

        # this constraint was introduced by us and not thr BMW paper
        # constraint 3
        """ensure that for each p is not combined with multiple others at once"""
        for p in self.data.P:
            q_indices = np.where(np.array(self.data.Psp) == int(p))[0]
            constr = sum(sp_variables[_h] for _h in q_indices)
            if build:
                self.__model.add_constraint(constr <= 1)
            constr_dict[f"const.3 p={p}"] = (constr, 1)

        """For platforms q that can be combined to a larger platform:
        the number of cars x_kp that are assigned to these subsets must
        be smaller than the number of elements of q, if the platforms 
        are not combined, or must be smaller than 1, in the case the 
        platforms are combined."""
        # constraint 4
        for q_idx, q in enumerate(self.data.Psp):
            constr = sum(
                [
                    x_variables[(k, p)]
                    for (k, p) in itertools.product(range(self.data.K), q)
                ]
            )
            max_constr = len(q) * (1 - sp_variables[q_idx]) + sp_variables[q_idx]
            if build:
                self.__model.add_constraint(constr <= max_constr)
            constr_dict[f"const.4 q_idx={q_idx}"] = (constr, max_constr)

        if not build:
            return constr_dict

    def length_constraints(
        self,
        x_variables,
        a_variables,
        sp_variables,
        d_variables,
        build=True,
    ):
        """
        Length constraints

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """

        constr_dict = {}
        """The total length of the cars on a platform can not be larger than the maximally allowed Length"""
        # constraint 5
        for L_idx, L in enumerate(self.data.Pl):
            L_set = set([i for i in L])

            angled_L = sum(
                [
                    x_variables[(k, p)] * self.data.vehicles[k]["Length"]
                    - self.data.vehicles[k]["Length"]
                    * x_variables[(k, p)] * a_variables[self.data.Pa.index(p)]
                    * (
                        self.data.lr_coefficient[self.data.vehicles[k]["Class"]][0][0]
                        * (1-d_variables[p]) * (1-d_variables[self.data.Pa_nu[p]])
                        + self.data.lr_coefficient[self.data.vehicles[k]["Class"]][0][1]
                        * (1-d_variables[p]) * d_variables[self.data.Pa_nu[p]]
                        + self.data.lr_coefficient[self.data.vehicles[k]["Class"]][1][0]
                        * d_variables[p] * (1-d_variables[self.data.Pa_nu[p]])
                        + self.data.lr_coefficient[self.data.vehicles[k]["Class"]][1][1]
                        * d_variables[p] * d_variables[self.data.Pa_nu[p]]
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
            constr_dict[f"const.5 L_idx={L_idx}"] = (constr, max_constr)

            if build:
                #print("build angled_L",angled_L)
                #print("build not_angled_L",not_angled_L)
                #print("build max_constr", max_constr)
                self.__model.add_constraint(constr <= max_constr)

        """If platforms are combined to a larger platform, none of the individual platforms can be angled"""
        # constraint 6
        for q_idx, q_ in enumerate(self.data.Psp):
            q_set = set(q_)
            constr = sum(
                a_variables[self.data.Pa.index(p)]
                for p in q_set.intersection(self.data.Pa_set)
            )
            max_constr = len(q_) * (1 - sp_variables[q_idx])
            constr_dict[f"const.6 q_idx={q_idx}"] = (constr, max_constr)

            if build:
                self.__model.add_constraint(constr <= max_constr)
        if not build:
            return constr_dict

    def height_constraint(
        self,
        x_variables,
        a_variables,
        d_variables,
        build=True,
    ):
        """
        Height constraints

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        constr_dict = {}
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
                    * x_variables[(k, p)] * a_variables[self.data.Pa.index(p)]
                    * (
                        self.data.hr_coefficient[self.data.vehicles[k]["Class"]][0][0]
                        * (1-d_variables[p]) * (1-d_variables[self.data.Pa_nu[p]])
                        + self.data.hr_coefficient[self.data.vehicles[k]["Class"]][0][1]
                        * (1-d_variables[p]) * d_variables[self.data.Pa_nu[p]]
                        + self.data.hr_coefficient[self.data.vehicles[k]["Class"]][1][0]
                        * d_variables[p] * (1-d_variables[self.data.Pa_nu[p]])
                        + self.data.hr_coefficient[self.data.vehicles[k]["Class"]][1][1]
                        * d_variables[p] * d_variables[self.data.Pa_nu[p]]
                    )
                    for (k, p) in itertools.product(
                        range(self.data.K), list(H_set.intersection(self.data.Pa_set))
                    )
                ]
            )
            constr = angled_height + non_angled_height
            max_constr = self.data.H_max[H_idx]
            constr_dict[f"const.7 H_idx={H_idx}"] = (constr, max_constr)
            if build:
                self.__model.add_constraint(constr <= max_constr)
        if not build:
            return constr_dict

    def weight_constraints(
        self, x_variables, sp_variables, a_variables, build=True
    ):
        """
        Weight constraints

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        constr_dict = {}

        """Platforms than can not be angled, have to be checked so that the weight
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
                constr_dict[f"const.8 p={p}"] = (weight, max_pltfrm_weight)
                if build:
                    self.__model.add_constraint(weight <= max_pltfrm_weight)

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
                constr_dict[f"const.9 (p, q_idx)={(p, q_idx)}"] = (
                    weight,
                    gamma_constraint,
                )

                if build:
                    self.__model.add_constraint(weight <= gamma_constraint)

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
                constr_dict[f"const.10 p={p}"] = (weight, max_angle_weight)

                if build:
                    self.__model.add_constraint(weight <= max_angle_weight)

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
                        x_variables[(k, p)] * a_variables[self.data.Pa.index(p)]
                        * (
                            sp_variables[q_idx] * self.data.wc_max[q_idx]
                            + (1 - sp_variables[q_idx]) * self.data.wp_max[p]
                        )
                        for k in range(self.data.K)
                    ]
                )
                max_sp_a_weight = -anlge_gamma_constr + angle_const + gamma_constraint
                constr_dict[f"const.11 (p, q_idx)={(p, q_idx)}"] = (
                    weight,
                    max_sp_a_weight,
                )

                if build:
                    self.__model.add_constraint(weight <= max_sp_a_weight)

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
            constr_dict[f"const.12 L_idx={L_idx}"] = (platform_weight, max_pl_weight)

            if build:
                self.__model.add_constraint(platform_weight <= max_pl_weight)

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
            constr_dict[f"const. 13 T_idx={T_idx}"] = (vehicle_weight, max_t_weight)

            if build:
                self.__model.add_constraint(vehicle_weight <= max_t_weight)

        """The total weight of the vehicles on all platforms must be smaller than W_max"""
        # constraint 14
        total_weight = sum(
            self.data.vehicles[k]["Weight"] * x_variables[(k, p)]
            for (k, p) in itertools.product(range(self.data.K), self.data.P)
        )
        max_T_weight = self.data.W_max
        constr_dict[f"const. 14"] = (total_weight, max_T_weight)

        if build:
            self.__model.add_constraint(total_weight <= max_T_weight)
        else:
            return constr_dict

    def allowed_variables(self):
        """
       Define the allowed indices for the different decision variables.

       Note: ax is a slack variable for linearisation of the quadratic terms a*x
       ,while dd is the slack variable for the linearisation of d*d variables
       , dd is only necessary for pa variables, since only quadratic d terms of
       angled platforms times their neighbour will exist in the formulation.

       All indices will always be starting from 0.
       """

        return {
            "x": [
                (k, p)
                for k, p in itertools.product(
                    range(self.data.K), range(len(self.data.P))
                )
            ],
            "a": [a for a in range(len(self.data.Pa))],
            "sp": [sp for sp in range(len(self.data.Psp))],
            "d": [d for d in range(len(self.data.P))],
            "ax": [
                (k, p)
                for k, p in itertools.product(
                    range(self.data.K), range(len(self.data.Pa))
                )
            ],
            "dd": [a for a in range(len(self.data.Pa))],
        }

    @property
    def variables(self):
        return self.__variables

    @property
    def model(self):
        return self.__model

    def check_solution_test(self, solution):
        # nochmal durchschauen das alle checks richtig sind nach dem shuffle der constraints
        x_variables = solution["x_variables"]
        a_variables = solution["a_variables"]
        sp_variables = solution["sp_variables"]
        d_variables = solution["d_variables"]


        all_constraints = {}
        equal_constr = {}

        assign_constraints = self.assignment_constraints(
            x_variables=x_variables, sp_variables=sp_variables, build=False
        )
        all_constraints.update(assign_constraints)

        length_constraints = self.length_constraints(
            x_variables=x_variables,
            a_variables=a_variables,
            sp_variables=sp_variables,
            d_variables=d_variables,
            build=False,
        )
        all_constraints.update(length_constraints)

        height_constraints = self.height_constraint(
            x_variables=x_variables,
            a_variables=a_variables,
            d_variables=d_variables,
            build=False,
        )
        all_constraints.update(height_constraints)

        weight_constraints = self.weight_constraints(
            x_variables=x_variables,
            sp_variables=sp_variables,
            a_variables=a_variables,
            build=False,
        )
        all_constraints.update(weight_constraints)

        for name, values in equal_constr.items():
            Delta = values[1] - values[0]
            #print(name, values[0], values[1])
            if Delta == 0:
                print("PASSED")
            else:
                print("Failed", Delta)

        for name, values in all_constraints.items():
            Delta = values[1] - values[0]
            #print(name, values[0], values[1])
            if Delta >= 0:
                print("PASSED")
            else:
                print("Failed", Delta)