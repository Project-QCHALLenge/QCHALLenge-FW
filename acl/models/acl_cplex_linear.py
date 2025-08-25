import itertools
import time
import numpy as np
from enum import Enum
from docplex.mp.model import Model as CplexModel

from acl.data.acl_data import ACLData

from transformations.from_cplex import FromCPLEX

class Sense(Enum):
    Min = "min"
    Max = "max"


class CplexLinACL(CplexModel):
    """
    Class to create a Cplex model, with maximally linear terms, for the Autocarrier Loading Problem
    """

    def __init__(self, data: ACLData, d_var_exists=False, old_sack_var=False):
        """
        Initialize the Quad_CplexACL class with ACLData.

        Parameters:
        - data (ACLData): Input data for the model. data includes, truck and vehicle parameters.
        """
        self.d_var_exists = d_var_exists
        self.old_sack_var = old_sack_var

        self.data: ACLData = data

        # define decision parameter indices and initialise the cplex model
        self.__allowed_variables = self.allowed_variables()
        self.__model = CplexModel(name="ACL_linear")

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
        if d_var_exists and not old_sack_var:
            self.__d_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["d"], name="d"
            )
            self.__d00_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["dd"], name="d00"
            )
            self.__d01_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["dd"], name="d01"
            )
            self.__d10_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["dd"], name="d10"
            )
            self.__d11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["dd"], name="d11"
            )
            # define the a-x-d-d variables
            self.__axdd30_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd30"
            )
            self.__axdd31_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd31"
            )
            self.__axdd32_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd32"
            )
            self.__axdd33_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd33"
            )
            ## a-x linear terms
            self.__ax11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["ax"], name="ax11"
            )
            ## x-sp linear terms
            self.__xsp11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["xsp"], name="xsp11"
            )
            ## a-x-sp linear terms
            self.__axsp31_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp31"
            )

            self.__variables = {
                "x_variables": self.__x_variables,
                "a_variables": self.__a_variables,
                "sp_variables": self.__sp_variables,
                "d_variables": self.__d_variables,
                "d00_variables": self.__d00_variables,
                "d01_variables": self.__d01_variables,
                "d10_variables": self.__d10_variables,
                "d11_variables": self.__d11_variables,
                "axdd30_variables": self.__axdd30_variables,
                "axdd31_variables": self.__axdd31_variables,
                "axdd32_variables": self.__axdd32_variables,
                "axdd33_variables": self.__axdd33_variables,
                "axsp31_variables": self.__axsp31_variables,
                "ax11_variables": self.__ax11_variables,
                "xsp11_variables": self.__xsp11_variables,
            }

        elif d_var_exists and old_sack_var:
            self.__d_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["d"], name="d"
            )
            # d-d linear terms
            self.__d00_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["dd"], name="d00"
            )
            self.__d01_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["dd"], name="d01"
            )
            self.__d10_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["dd"], name="d10"
            )
            self.__d11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["dd"], name="d11"
            )

            # define the 16 a-x-d-d variables
            self.__axdd00_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd00"
            )
            self.__axdd01_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd01"
            )
            self.__axdd02_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd02"
            )
            self.__axdd03_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd03"
            )

            self.__axdd10_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd10"
            )
            self.__axdd11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd11"
            )
            self.__axdd12_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd12"
            )
            self.__axdd13_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd13"
            )

            self.__axdd20_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd20"
            )
            self.__axdd21_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd21"
            )
            self.__axdd22_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd22"
            )
            self.__axdd23_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd23"
            )

            self.__axdd30_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd30"
            )
            self.__axdd31_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd31"
            )
            self.__axdd32_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd32"
            )
            self.__axdd33_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axdd"], name="axdd33"
            )

            ## a-x linear terms
            self.__ax00_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["ax"], name="ax00"
            )
            self.__ax01_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["ax"], name="ax01"
            )
            self.__ax10_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["ax"], name="ax10"
            )
            self.__ax11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["ax"], name="ax11"
            )
            ## x-sp linear terms
            self.__xsp00_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["xsp"], name="xsp00"
            )
            self.__xsp01_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["xsp"], name="xsp01"
            )
            self.__xsp10_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["xsp"], name="xsp10"
            )
            self.__xsp11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["xsp"], name="xsp11"
            )
            ## define the 8 a-x-sp variables
            self.__axsp00_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp00"
            )
            self.__axsp01_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp01"
            )
            self.__axsp10_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp10"
            )
            self.__axsp11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp11"
            )
            self.__axsp20_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp20"
            )
            self.__axsp21_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp21"
            )
            self.__axsp30_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp30"
            )
            self.__axsp31_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp31"
            )

            self.__variables = {
                "x_variables": self.__x_variables,
                "a_variables": self.__a_variables,
                "sp_variables": self.__sp_variables,
                "d_variables": self.__d_variables,
                "d00_variables": self.__d00_variables,
                "d01_variables": self.__d01_variables,
                "d10_variables": self.__d10_variables,
                "d11_variables": self.__d11_variables,
                "ax00_variables": self.__ax00_variables,
                "ax01_variables": self.__ax01_variables,
                "ax10_variables": self.__ax10_variables,
                "ax11_variables": self.__ax11_variables,
                "xsp00_variables": self.__xsp00_variables,
                "xsp01_variables": self.__xsp01_variables,
                "xsp10_variables": self.__xsp10_variables,
                "xsp11_variables": self.__xsp11_variables,
                "axsp00_variables": self.__axsp00_variables,
                "axsp01_variables": self.__axsp01_variables,
                "axsp10_variables": self.__axsp10_variables,
                "axsp11_variables": self.__axsp11_variables,
                "axsp20_variables": self.__axsp20_variables,
                "axsp21_variables": self.__axsp21_variables,
                "axsp30_variables": self.__axsp30_variables,
                "axsp31_variables": self.__axsp31_variables,
                "axdd00_variables": self.__axdd00_variables,
                "axdd01_variables": self.__axdd01_variables,
                "axdd02_variables": self.__axdd02_variables,
                "axdd03_variables": self.__axdd03_variables,
                "axdd10_variables": self.__axdd10_variables,
                "axdd11_variables": self.__axdd11_variables,
                "axdd12_variables": self.__axdd12_variables,
                "axdd13_variables": self.__axdd13_variables,
                "axdd20_variables": self.__axdd20_variables,
                "axdd21_variables": self.__axdd21_variables,
                "axdd22_variables": self.__axdd22_variables,
                "axdd23_variables": self.__axdd23_variables,
                "axdd30_variables": self.__axdd30_variables,
                "axdd31_variables": self.__axdd31_variables,
                "axdd32_variables": self.__axdd32_variables,
                "axdd33_variables": self.__axdd33_variables,
            }

        elif not d_var_exists and old_sack_var:
            ## a-x linear terms
            self.__ax00_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["ax"], name="ax00"
            )
            self.__ax01_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["ax"], name="ax01"
            )
            self.__ax10_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["ax"], name="ax10"
            )
            self.__ax11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["ax"], name="ax11"
            )
            ## x-sp linear terms
            self.__xsp00_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["xsp"], name="xsp00"
            )
            self.__xsp01_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["xsp"], name="xsp01"
            )
            self.__xsp10_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["xsp"], name="xsp10"
            )
            self.__xsp11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["xsp"], name="xsp11"
            )
            ## define the 8 a-x-sp variables
            self.__axsp00_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp00"
            )
            self.__axsp01_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp01"
            )
            self.__axsp10_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp10"
            )
            self.__axsp11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp11"
            )
            self.__axsp20_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp20"
            )
            self.__axsp21_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp21"
            )
            self.__axsp30_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp30"
            )
            self.__axsp31_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp31"
            )

            self.__variables = {
                "x_variables": self.__x_variables,
                "a_variables": self.__a_variables,
                "sp_variables": self.__sp_variables,
                "ax00_variables": self.__ax00_variables,
                "ax01_variables": self.__ax01_variables,
                "ax10_variables": self.__ax10_variables,
                "ax11_variables": self.__ax11_variables,
                "xsp00_variables": self.__xsp00_variables,
                "xsp01_variables": self.__xsp01_variables,
                "xsp10_variables": self.__xsp10_variables,
                "xsp11_variables": self.__xsp11_variables,
                "axsp00_variables": self.__axsp00_variables,
                "axsp01_variables": self.__axsp01_variables,
                "axsp10_variables": self.__axsp10_variables,
                "axsp11_variables": self.__axsp11_variables,
                "axsp20_variables": self.__axsp20_variables,
                "axsp21_variables": self.__axsp21_variables,
                "axsp30_variables": self.__axsp30_variables,
                "axsp31_variables": self.__axsp31_variables,
            }

        elif not d_var_exists and not old_sack_var:
            ## a-x linear terms
            self.__ax11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["ax"], name="ax11"
            )
            ## x-sp linear terms
            self.__xsp11_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["xsp"], name="xsp11"
            )
            ## a-x-sp linear terms
            self.__axsp31_variables = self.__model.binary_var_dict(
                keys=self.__allowed_variables["axsp"], name="axsp31"
            )

            self.__variables = {
                "x_variables": self.__x_variables,
                "a_variables": self.__a_variables,
                "sp_variables": self.__sp_variables,
                "ax11_variables": self.__ax11_variables,
                "xsp11_variables": self.__xsp11_variables,
                "axsp31_variables": self.__axsp31_variables,
            }

        self.build_model()

    def build_model(self):
        """
        Build the model
        """
        # objective function
        self.objective_function(variable=self.__x_variables)

        # assignement constraints
        self.assignment_constraints(
            x_variables=self.__x_variables, sp_variables=self.__sp_variables
        )

        # weight constraints
        self.weight_constraints(
            x_variables=self.__x_variables,
            a_variables=self.__a_variables,
            ax11_variables=self.__ax11_variables,
            xsp11_variables=self.__xsp11_variables,
            axsp31_variables=self.__axsp31_variables,
        )
        # slack variable constraints
        if self.d_var_exists and self.old_sack_var:
            self.dd_slack_variables(
                d_variable=self.__d_variables,
                d00_variables=self.__d00_variables,
                d01_variables=self.__d01_variables,
                d10_variables=self.__d10_variables,
                d11_variables=self.__d11_variables,
            )

            self.ax_dd_slack_variables_old(
                d00_variables=self.__d00_variables,
                d01_variables=self.__d01_variables,
                d10_variables=self.__d10_variables,
                d11_variables=self.__d11_variables,
                ax00_variables=self.__ax00_variables,
                ax01_variables=self.__ax01_variables,
                ax10_variables=self.__ax10_variables,
                ax11_variables=self.__ax11_variables,
                axdd00_variables=self.__axdd00_variables,
                axdd01_variables=self.__axdd01_variables,
                axdd02_variables=self.__axdd02_variables,
                axdd03_variables=self.__axdd03_variables,
                axdd10_variables=self.__axdd10_variables,
                axdd11_variables=self.__axdd11_variables,
                axdd12_variables=self.__axdd12_variables,
                axdd13_variables=self.__axdd13_variables,
                axdd20_variables=self.__axdd20_variables,
                axdd21_variables=self.__axdd21_variables,
                axdd22_variables=self.__axdd22_variables,
                axdd23_variables=self.__axdd23_variables,
                axdd30_variables=self.__axdd30_variables,
                axdd31_variables=self.__axdd31_variables,
                axdd32_variables=self.__axdd32_variables,
                axdd33_variables=self.__axdd33_variables,
            )

            self.ax_slack_variables_old(
                a_variables=self.__a_variables,
                x_variables=self.__x_variables,
                ax00_variables=self.__ax00_variables,
                ax01_variables=self.__ax01_variables,
                ax10_variables=self.__ax10_variables,
                ax11_variables=self.__ax11_variables,
            )

            self.xsp_slack_variable_old(
                x_variables=self.__x_variables,
                sp_variables=self.__sp_variables,
                xsp00_variables=self.__xsp00_variables,
                xsp01_variables=self.__xsp01_variables,
                xsp10_variables=self.__xsp10_variables,
                xsp11_variables=self.__xsp11_variables,
            )

            self.axsp_slack_variables_old(
                sp_variables=self.__sp_variables,
                ax00_variables=self.__ax00_variables,
                ax01_variables=self.__ax01_variables,
                ax10_variables=self.__ax10_variables,
                ax11_variables=self.__ax11_variables,
                axsp00_variables=self.__axsp00_variables,
                axsp01_variables=self.__axsp01_variables,
                axsp10_variables=self.__axsp10_variables,
                axsp11_variables=self.__axsp11_variables,
                axsp20_variables=self.__axsp20_variables,
                axsp21_variables=self.__axsp21_variables,
                axsp30_variables=self.__axsp30_variables,
                axsp31_variables=self.__axsp31_variables,
            )

            # length constraints
            self.length_constraints(
                x_variables=self.__x_variables,
                a_variables=self.__a_variables,
                sp_variables=self.__sp_variables,
                axdd30_variables=self.__axdd30_variables,
                axdd31_variables=self.__axdd31_variables,
                axdd32_variables=self.__axdd32_variables,
                axdd33_variables=self.__axdd33_variables,
            )
            # height constraints
            self.height_constraint(
                x_variables=self.__x_variables,
                axdd30_variables=self.__axdd30_variables,
                axdd31_variables=self.__axdd31_variables,
                axdd32_variables=self.__axdd32_variables,
                axdd33_variables=self.__axdd33_variables,
            )

        if self.d_var_exists and not self.old_sack_var:
            self.dd_slack_variables(
                d_variable=self.__d_variables,
                d00_variables=self.__d00_variables,
                d01_variables=self.__d01_variables,
                d10_variables=self.__d10_variables,
                d11_variables=self.__d11_variables,
            )

            self.ax_dd_slack_variables(
                d00_variables=self.__d00_variables,
                d01_variables=self.__d01_variables,
                d10_variables=self.__d10_variables,
                d11_variables=self.__d11_variables,
                ax11_variables=self.__ax11_variables,
                axdd30_variables=self.__axdd30_variables,
                axdd31_variables=self.__axdd31_variables,
                axdd32_variables=self.__axdd32_variables,
                axdd33_variables=self.__axdd33_variables,
            )

            self.ax_slack_variables(
                a_variables=self.__a_variables,
                x_variables=self.__x_variables,
                ax11_variables=self.__ax11_variables,
            )

            self.xsp_slack_variable(
                x_variables=self.__x_variables,
                sp_variables=self.__sp_variables,
                xsp11_variables=self.__xsp11_variables,
            )

            self.axsp_slack_variables(
                sp_variables=self.__sp_variables,
                ax11_variables=self.__ax11_variables,
                axsp31_variables=self.__axsp31_variables,
            )
            # length constraints
            self.length_constraints(
                x_variables=self.__x_variables,
                a_variables=self.__a_variables,
                sp_variables=self.__sp_variables,
                axdd30_variables=self.__axdd30_variables,
                axdd31_variables=self.__axdd31_variables,
                axdd32_variables=self.__axdd32_variables,
                axdd33_variables=self.__axdd33_variables,
            )
            # height constraints
            self.height_constraint(
                x_variables=self.__x_variables,
                axdd30_variables=self.__axdd30_variables,
                axdd31_variables=self.__axdd31_variables,
                axdd32_variables=self.__axdd32_variables,
                axdd33_variables=self.__axdd33_variables,
            )

        if not self.d_var_exists and self.old_sack_var:
            self.ax_slack_variables_old(
                a_variables=self.__a_variables,
                x_variables=self.__x_variables,
                ax00_variables=self.__ax00_variables,
                ax01_variables=self.__ax01_variables,
                ax10_variables=self.__ax10_variables,
                ax11_variables=self.__ax11_variables,
            )

            self.xsp_slack_variable_old(
                x_variables=self.__x_variables,
                sp_variables=self.__sp_variables,
                xsp00_variables=self.__xsp00_variables,
                xsp01_variables=self.__xsp01_variables,
                xsp10_variables=self.__xsp10_variables,
                xsp11_variables=self.__xsp11_variables,
            )

            self.axsp_slack_variables_old(
                sp_variables=self.__sp_variables,
                ax00_variables=self.__ax00_variables,
                ax01_variables=self.__ax01_variables,
                ax10_variables=self.__ax10_variables,
                ax11_variables=self.__ax11_variables,
                axsp00_variables=self.__axsp00_variables,
                axsp01_variables=self.__axsp01_variables,
                axsp10_variables=self.__axsp10_variables,
                axsp11_variables=self.__axsp11_variables,
                axsp20_variables=self.__axsp20_variables,
                axsp21_variables=self.__axsp21_variables,
                axsp30_variables=self.__axsp30_variables,
                axsp31_variables=self.__axsp31_variables,
            )

            # length constraints
            self.length_constraints_no_d(
                x_variables=self.__x_variables,
                a_variables=self.__a_variables,
                sp_variables=self.__sp_variables,
                ax11_variables=self.__ax11_variables,
            )
            # height constraints
            self.height_constraint_no_d(
                x_variables=self.__x_variables, ax11_variables=self.__ax11_variables
            )

        if not self.d_var_exists and not self.old_sack_var:
            self.ax_slack_variables(
                a_variables=self.__a_variables,
                x_variables=self.__x_variables,
                ax11_variables=self.__ax11_variables,
            )

            self.xsp_slack_variable(
                x_variables=self.__x_variables,
                sp_variables=self.__sp_variables,
                xsp11_variables=self.__xsp11_variables,
            )

            self.axsp_slack_variables(
                sp_variables=self.__sp_variables,
                ax11_variables=self.__ax11_variables,
                axsp31_variables=self.__axsp31_variables,
            )

            # length constraints
            self.length_constraints_no_d(
                x_variables=self.__x_variables,
                a_variables=self.__a_variables,
                sp_variables=self.__sp_variables,
                ax11_variables=self.__ax11_variables,
            )
            # height constraints
            self.height_constraint_no_d(
                x_variables=self.__x_variables, ax11_variables=self.__ax11_variables
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

    def solve_qubo(self, solve_func, **config):

        lgrng_multplr = self.get_default_lagrange_multiplier(multiplicative_factor=config.get("lagrange_mult_multiplicative_factor"))

        qubo, converter = FromCPLEX(self.__model).to_matrix(lagrange_multiplier=lgrng_multplr)

        if "lagrange_mult_multiplicative_factor" in config:
            del config["lagrange_mult_multiplicative_factor"]

        start_time = time.time()
        answer = solve_func(Q=qubo, **config)
        runtime = time.time() - start_time

        solution = converter(answer.first.sample)

        return {"solution": solution, "energy": answer.first.energy, "runtime": runtime}

    def get_default_lagrange_multiplier(self, multiplicative_factor=1):
        """
        Extracts the largest (linear) bias of the model, since the objective function does not have quadratic terms, in
        order to find the default lagrange multiplier which is 10*max(bias)
        """
        objective = self.__model.get_objective_expr()

        linear_biases = [
            -objective.get_coef(var) for var in self.__model.iter_variables()
        ]

        if multiplicative_factor is None:
            return None
        else:
            return 10 * max(linear_biases) * multiplicative_factor

    def objective_function(self, variable, build=True):
        """
        Define objective function.
        Returns: the objective value if build=False and sets the maximisation objective for self.__moedel if build=True
        """

        obj_func = -sum(
            [
                variable[(k, p)]*1000
                for (k, p) in itertools.product(
                    range(self.data.K), range(len(self.data.P))
                )
            ]
        )
        if build:
            self.__model.set_objective(sense=Sense.Min.value, expr=obj_func)
        else:
            return obj_func

    def dd_slack_variables(
        self,
        d_variable,
        d00_variables,
        d01_variables,
        d10_variables,
        d11_variables,
        build=True,
    ):
        """
        Define dd slack variables

        The slack variable:
        d00[idx_p]==1 if d[p]==0 and d[nu_p]==0 (the neighbouring car)
        d01[idx_p]==1 if d[p]==0 and d[nu_p]==1 (the neighbouring car)
        d10[idx_p]==1 if d[p]==1 and d[nu_p]==0 (the neighbouring car)
        d11[idx_p]==1 if d[p]==1 and d[nu_p]==1 (the neighbouring car)

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        constr_dict = {}
        equal_constr_dict = {}

        for idx_p, p in enumerate(self.data.Pa):
            constr_00 = (1 - d_variable[p]) + (1 - d_variable[self.data.Pa_nu[idx_p]])
            constr_01 = (1 - d_variable[p]) + d_variable[self.data.Pa_nu[idx_p]]
            constr_10 = d_variable[p] + (1 - d_variable[self.data.Pa_nu[idx_p]])
            constr_11 = d_variable[p] + d_variable[self.data.Pa_nu[idx_p]]
            total_dd = (
                d00_variables[idx_p]
                + d01_variables[idx_p]
                + d10_variables[idx_p]
                + d11_variables[idx_p]
            )

            constr_dict[f"dd_00_p{p}"] = (2 * d00_variables[idx_p], constr_00)
            constr_dict[f"dd_01_p{p}"] = (2 * d01_variables[idx_p], constr_01)
            constr_dict[f"dd_10_p{p}"] = (2 * d10_variables[idx_p], constr_10)
            constr_dict[f"dd_11_p{p}"] = (2 * d11_variables[idx_p], constr_11)
            # changed order of the tupel as in this inequality the inequaltiy is switched arond the opposite way
            equal_constr_dict[f"dd=1_p{p}"] = (total_dd, 1)

            if build:
                self.__model.add_constraint(constr_00 >= 2 * d00_variables[idx_p])  # 00
                self.__model.add_constraint(constr_01 >= 2 * d01_variables[idx_p])  # 01
                self.__model.add_constraint(constr_10 >= 2 * d10_variables[idx_p])  # 10
                self.__model.add_constraint(constr_11 >= 2 * d11_variables[idx_p])  # 11

                self.__model.add_constraint(total_dd == 1)

        if not build:
            return constr_dict, equal_constr_dict

    def ax_slack_variables(
        self,
        a_variables,
        x_variables,
        ax11_variables,
        build=True,
    ):
        """
        Define ax slack variables

        The slack variable:
        ax11[k,p]==1 if a[p]==1 and x[k,p]==1
        ax11[k,p] >= x[k,p] + a[p] -1    enforces ax==1 if both are equal 1
        ax11[k,p] <= x[k,p]              enforces ax==0 if x is equal 0
        ax11[k,p] <= a[p]                enforces ax==0 if a is equal 0
        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """

        constr_dict = {}
        equal_constr_dict = {}
        for k, p in self.__allowed_variables["ax"]:
            constr_11 = a_variables[p] + x_variables[(k, self.data.Pa[p])] - 1

            constr_dict[f"ax_11_k,p-a{k, p}"] = (ax11_variables[(k, p)], a_variables[p])
            constr_dict[f"ax_11_k,p-x{k, p}"] = (
                ax11_variables[(k, p)],
                x_variables[(k, self.data.Pa[p])],
            )
            constr_dict[f"ax_11_k,p{k, p}"] = (constr_11, ax11_variables[(k, p)])

            if build:
                self.__model.add_constraint(ax11_variables[(k, p)] >= constr_11)  # 11
                self.__model.add_constraint(a_variables[p] >= ax11_variables[(k, p)])
                self.__model.add_constraint(
                    x_variables[(k, self.data.Pa[p])] >= ax11_variables[(k, p)]
                )

        if not build:
            return constr_dict, equal_constr_dict

    def ax_slack_variables_old(
        self,
        a_variables,
        x_variables,
        ax00_variables,
        ax01_variables,
        ax10_variables,
        ax11_variables,
        build=True,
    ):
        """
        Define ax slack variables

        The slack variable:
        ax00[k,p]==1 if a[p]==0 and x[k,p]==0 (the neighbouring car)
        ax01[k,p]==1 if a[p]==0 and x[k,p]==1 (the neighbouring car)
        ax10[k,p]==1 if a[p]==1 and x[k,p]==0 (the neighbouring car)
        ax11[k,p]==1 if a[p]==1 and x[k,p]==1 (the neighbouring car)

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """

        constr_dict = {}
        equal_constr_dict = {}
        for k, p in self.__allowed_variables["ax"]:
            constr_00 = (1 - a_variables[p]) + (1 - x_variables[(k, self.data.Pa[p])])
            constr_01 = (1 - a_variables[p]) + x_variables[(k, self.data.Pa[p])]
            constr_10 = a_variables[p] + (1 - x_variables[(k, self.data.Pa[p])])
            constr_11 = a_variables[p] + x_variables[(k, self.data.Pa[p])]
            total_ax = (
                ax00_variables[(k, p)]
                + ax01_variables[(k, p)]
                + ax10_variables[(k, p)]
                + ax11_variables[(k, p)]
            )
            constr_dict[f"ax_00_k,p{k, p}"] = (2 * ax00_variables[(k, p)], constr_00)
            constr_dict[f"ax_01_k,p{k, p}"] = (2 * ax01_variables[(k, p)], constr_01)
            constr_dict[f"ax_10_k,p{k, p}"] = (2 * ax10_variables[(k, p)], constr_10)
            constr_dict[f"ax_11_k,p{k, p}"] = (2 * ax11_variables[(k, p)], constr_11)
            # changed order of tupel here same way as in the self.__length_constraints() method
            equal_constr_dict[f"ax_11_k,p{k, p}"] = (total_ax, 1)

            if build:
                self.__model.add_constraint(
                    constr_00 >= 2 * ax00_variables[(k, p)]
                )  # 00
                self.__model.add_constraint(
                    constr_01 >= 2 * ax01_variables[(k, p)]
                )  # 01
                self.__model.add_constraint(
                    constr_10 >= 2 * ax10_variables[(k, p)]
                )  # 10
                self.__model.add_constraint(
                    constr_11 >= 2 * ax11_variables[(k, p)]
                )  # 11

                self.__model.add_constraint(total_ax == 1)

        if not build:
            return constr_dict, equal_constr_dict

    def xsp_slack_variable(
        self,
        x_variables,
        sp_variables,
        xsp11_variables,
        build=True,
    ):
        """
        Define xsp slack variables

        The slack variable:
        xsp11[k,p,q]==1 if x[k,p]==1 and sp[q]==1
        xsp11[k,p,q] >= x[k,p] + sp[p] -1    enforces xsp==1 if both are equal 1
        xsp11[k,p,q] <= x[k,p]              enforces xsp==0 if x is equal 0
        xsp11[k,p,q] <= sp[q]               enforces xsp==0 if sp is equal 0

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        constr_dict = {}
        equal_constr_dict = {}
        for k, p, q in self.__allowed_variables["xsp"]:
            constr_11 = x_variables[(k, p)] + sp_variables[q] - 1

            constr_dict[f"new - xsp_11_k,p,q-x{(k, p, q)}"] = (
                xsp11_variables[(k, p, q)],
                x_variables[(k, p)],
            )
            constr_dict[f"new - xsp_11_k,p,q-sp{(k, p, q)}"] = (
                xsp11_variables[(k, p, q)],
                sp_variables[q],
            )
            constr_dict[f"new - xsp_11_k,p,q{(k, p, q)}"] = (
                constr_11,
                xsp11_variables[(k, p, q)],
            )

            if build:
                self.__model.add_constraint(
                    x_variables[(k, p)] >= xsp11_variables[(k, p, q)]
                )
                self.__model.add_constraint(
                    sp_variables[q] >= xsp11_variables[(k, p, q)]
                )
                self.__model.add_constraint(xsp11_variables[(k, p, q)] >= constr_11)

        if not build:
            return constr_dict, equal_constr_dict

    def xsp_slack_variable_old(
        self,
        x_variables,
        sp_variables,
        xsp00_variables,
        xsp01_variables,
        xsp10_variables,
        xsp11_variables,
        build=True,
    ):
        """
        Define xsp slack variables

        The slack variable:
        xsp00[k,p,q]==1 if x[k,p]==0 and sp[q]==0
        xsp01[k,p,q]==1 if x[k,p]==0 and sp[q]==1
        xsp10[k,p,q]==1 if x[k,p]==1 and sp[q]==0
        xsp11[k,p,q]==1 if x[k,p]==1 and sp[q]==1

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        constr_dict = {}
        equal_constr_dict = {}
        for k, p, q in self.__allowed_variables["xsp"]:
            constr_00 = (1 - x_variables[(k, p)]) + (1 - sp_variables[q])
            constr_01 = (1 - x_variables[(k, p)]) + sp_variables[q]
            constr_10 = x_variables[(k, p)] + (1 - sp_variables[q])
            constr_11 = x_variables[(k, p)] + sp_variables[q]
            total_xsp = (
                xsp00_variables[(k, p, q)]
                + xsp01_variables[(k, p, q)]
                + xsp10_variables[(k, p, q)]
                + xsp11_variables[(k, p, q)]
            )

            constr_dict[f"xsp_00_k,p,q{(k, p, q)}"] = (
                2 * xsp00_variables[(k, p, q)],
                constr_00,
            )
            constr_dict[f"xsp_01_k,p,q{(k, p, q)}"] = (
                2 * xsp01_variables[(k, p, q)],
                constr_01,
            )
            constr_dict[f"xsp_10_k,p,q{(k, p, q)}"] = (
                2 * xsp10_variables[(k, p, q)],
                constr_10,
            )
            constr_dict[f"xsp_11_k,p,q{(k, p, q)}"] = (
                2 * xsp11_variables[(k, p, q)],
                constr_11,
            )
            # changed order of tupel here same way as in the self.__length_constraints() method
            equal_constr_dict[f"xsp=1_k,p,q{(k, p, q)}"] = (total_xsp, 1)

            if build:
                self.__model.add_constraint(
                    constr_00 >= 2 * xsp00_variables[(k, p, q)]
                )  # 00
                self.__model.add_constraint(
                    constr_01 >= 2 * xsp01_variables[(k, p, q)]
                )  # 01
                self.__model.add_constraint(
                    constr_10 >= 2 * xsp10_variables[(k, p, q)]
                )  # 10
                self.__model.add_constraint(
                    constr_11 >= 2 * xsp11_variables[(k, p, q)]
                )  # 11

                self.__model.add_constraint(total_xsp == 1)

        if not build:
            return constr_dict, equal_constr_dict

    def axsp_slack_variables(
        self, sp_variables, ax11_variables, axsp31_variables, build=True
    ):
        """
        Define xsp slack variables

        The slack variable:
        axsp31[k,p,q]==1 if ax11[k,p]==1 and sp[q]==1
        axsp31[k,p,q] >= ax[k,p] + sp[q] -1    enforces axsp==1 if both are equal 1
        axsp31[k,p,q] <= x[k,p]                enforces axsp==0 if x is equal 0
        axsp31[k,p,q] <= sp[q]                 enforces axsp==0 if sp is equal 0

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """

        constr_dict = {}
        equal_constr_dict = {}
        for k, p, q in self.__allowed_variables["axsp"]:
            constr_31 = ax11_variables[(k, self.data.Pa.index(p))] + sp_variables[q] - 1

            constr_dict[f"new - axsp_31_k,p,q-ax11{(k, p, q)}"] = (
                axsp31_variables[(k, p, q)],
                ax11_variables[(k, self.data.Pa.index(p))],
            )
            constr_dict[f"new - axsp_31_k,p,q-sp{(k, p, q)}"] = (
                axsp31_variables[(k, p, q)],
                sp_variables[q],
            )
            constr_dict[f"new - axsp_31_k,p,q{(k, p, q)}"] = (
                constr_31,
                axsp31_variables[(k, p, q)],
            )

            if build:
                self.__model.add_constraint(
                    ax11_variables[(k, self.data.Pa.index(p))]
                    >= axsp31_variables[(k, p, q)]
                )

                self.__model.add_constraint(
                    sp_variables[q] >= axsp31_variables[(k, p, q)]
                )

                self.__model.add_constraint(axsp31_variables[(k, p, q)] >= constr_31)

        if not build:
            return constr_dict, equal_constr_dict

    def axsp_slack_variables_old(
        self,
        sp_variables,
        ax00_variables,
        ax01_variables,
        ax10_variables,
        ax11_variables,
        axsp00_variables,
        axsp01_variables,
        axsp10_variables,
        axsp11_variables,
        axsp20_variables,
        axsp21_variables,
        axsp30_variables,
        axsp31_variables,
        build=True,
    ):
        """
        Define xsp slack variables

        The slack variable:
        axsp00[k,p,q]==1 if ax00[k,p]==1 and sp[q]==0
        axsp01[k,p,q]==1 if ax00[k,p]==1 and sp[q]==1
        axsp10[k,p,q]==1 if ax01[k,p]==1 and sp[q]==0
        axsp11[k,p,q]==1 if ax01[k,p]==1 and sp[q]==1
        axsp20[k,p,q]==1 if ax10[k,p]==1 and sp[q]==0
        axsp21[k,p,q]==1 if ax10[k,p]==1 and sp[q]==1
        axsp30[k,p,q]==1 if ax11[k,p]==1 and sp[q]==0
        axsp31[k,p,q]==1 if ax11[k,p]==1 and sp[q]==1

        Note: Due to the constraint on the ax slack variables, one of the slack variables always
        has to be equal to one. Since all ax slack variables are tested here, each with
        the (1-sp) and sp probability,thereby one of the below instances has to be equal to 1

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        constr_dict = {}
        equal_constr_dict = {}
        for k, p, q in self.__allowed_variables["axsp"]:
            constr_00 = ax00_variables[(k, self.data.Pa.index(p))] + (
                1 - sp_variables[q]
            )
            constr_01 = ax00_variables[(k, self.data.Pa.index(p))] + sp_variables[q]
            constr_10 = ax01_variables[(k, self.data.Pa.index(p))] + (
                1 - sp_variables[q]
            )
            constr_11 = ax01_variables[(k, self.data.Pa.index(p))] + sp_variables[q]
            constr_20 = ax10_variables[(k, self.data.Pa.index(p))] + (
                1 - sp_variables[q]
            )
            constr_21 = ax10_variables[(k, self.data.Pa.index(p))] + sp_variables[q]
            constr_30 = ax11_variables[(k, self.data.Pa.index(p))] + (
                1 - sp_variables[q]
            )
            constr_31 = ax11_variables[(k, self.data.Pa.index(p))] + sp_variables[q]

            total_axsp = (
                axsp00_variables[(k, p, q)]
                + axsp01_variables[(k, p, q)]
                + axsp10_variables[(k, p, q)]
                + axsp11_variables[(k, p, q)]
                + axsp20_variables[(k, p, q)]
                + axsp21_variables[(k, p, q)]
                + axsp30_variables[(k, p, q)]
                + axsp31_variables[(k, p, q)]
            )

            constr_dict[f"axsp_00_k,p{(k, p, q)}"] = (
                2 * axsp00_variables[(k, p, q)],
                constr_00,
            )
            constr_dict[f"axsp_01_k,p{(k, p, q)}"] = (
                2 * axsp01_variables[(k, p, q)],
                constr_01,
            )
            constr_dict[f"axsp_10_k,p{(k, p, q)}"] = (
                2 * axsp10_variables[(k, p, q)],
                constr_10,
            )
            constr_dict[f"axsp_11_k,p{(k, p, q)}"] = (
                2 * axsp11_variables[(k, p, q)],
                constr_11,
            )
            constr_dict[f"axsp_20_k,p{(k, p, q)}"] = (
                2 * axsp20_variables[(k, p, q)],
                constr_20,
            )
            constr_dict[f"axsp_21_k,p{(k, p, q)}"] = (
                2 * axsp21_variables[(k, p, q)],
                constr_21,
            )
            constr_dict[f"axsp_30_k,p{(k, p, q)}"] = (
                2 * axsp30_variables[(k, p, q)],
                constr_30,
            )
            constr_dict[f"axsp_31_k,p{(k, p, q)}"] = (
                2 * axsp31_variables[(k, p, q)],
                constr_31,
            )
            # changed order of tupel here same way as in the self.__length_constraints() method
            equal_constr_dict[f"axsp=1_k,p{k, p}"] = (total_axsp, 1)
            if build:
                self.__model.add_constraint(
                    constr_00 >= 2 * axsp00_variables[(k, p, q)]
                )  # 00

                self.__model.add_constraint(
                    constr_01 >= 2 * axsp01_variables[(k, p, q)]
                )  # 01

                self.__model.add_constraint(
                    constr_10 >= 2 * axsp10_variables[(k, p, q)]
                )  # 10

                self.__model.add_constraint(
                    constr_11 >= 2 * axsp11_variables[(k, p, q)]
                )  # 11

                self.__model.add_constraint(
                    constr_20 >= 2 * axsp20_variables[(k, p, q)]
                )  # 20

                self.__model.add_constraint(
                    constr_21 >= 2 * axsp21_variables[(k, p, q)]
                )  # 21

                self.__model.add_constraint(
                    constr_30 >= 2 * axsp30_variables[(k, p, q)]
                )  # 30

                self.__model.add_constraint(
                    constr_31 >= 2 * axsp31_variables[(k, p, q)]
                )  # 31

                self.__model.add_constraint(total_axsp == 1)
        if not build:
            return constr_dict, equal_constr_dict

    def ax_dd_slack_variables(
        self,
        d00_variables,
        d01_variables,
        d10_variables,
        d11_variables,
        ax11_variables,
        axdd30_variables,
        axdd31_variables,
        axdd32_variables,
        axdd33_variables,
        build=True,
    ):
        """
        Define xsp slack variables

        The slack variable:
        axdd30[k,p,q]==1 if ax11[k,p]==1 and dd00[p]==1
        axdd30[k,p,q] >= ax11[k,p] + dd00[p] -1    enforces axdd30==1 if both are equal 1
        axdd30[k,p,q] <= ax11[k,p]                 enforces axdd30==0 if ax11 is equal 0
        axdd30[k,p,q] <= dd00[p]                   enforces axdd30==0 if dd00 is equal 0

        axdd31[k,p,q]==1 if ax11[k,p]==1 and dd01[p]==1
        axdd31[k,p,q] >= ax11[k,p] + dd01[p] -1    enforces axdd31==1 if both are equal 1
        axdd31[k,p,q] <= ax11[k,p]                 enforces axdd31==0 if ax11 is equal 0
        axdd31[k,p,q] <= dd01[p]                   enforces axdd31==0 if dd01 is equal 0

        axdd32[k,p,q]==1 if ax11[k,p]==1 and dd10[p]==1
        axdd32[k,p,q] >= ax11[k,p] + dd10[p] -1    enforces axdd32==1 if both are equal 1
        axdd32[k,p,q] <= ax11[k,p]                 enforces axdd32==0 if ax11 is equal 0
        axdd32[k,p,q] <= dd10[p]                   enforces axdd32==0 if dd10 is equal 0

        axdd33[k,p,q]==1 if ax11[k,p]==1 and dd11[p]==1
        axdd33[k,p,q] >= ax11[k,p] + dd11[p] -1    enforces axdd33==1 if both are equal 1
        axdd33[k,p,q] <= ax11[k,p]                 enforces axdd33==0 if ax11 is equal 0
        axdd33[k,p,q] <= dd11[p]                   enforces axdd33==0 if dd11 is equal 0

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """

        constr_dict = {}
        equal_constr_dict = {}
        for k, p in self.__allowed_variables["axdd"]:
            # the dd_slack_constraints ensure that only one of d00 d01 d10 d11. Thereby we can rest assured that also
            # at most only one (if ax11=1) of the axdd30 axdd31 axdd32 axdd33 variables will be one.
            constr_30 = ax11_variables[(k, p)] + d00_variables[p] - 1
            constr_31 = ax11_variables[(k, p)] + d01_variables[p] - 1
            constr_32 = ax11_variables[(k, p)] + d10_variables[p] - 1
            constr_33 = ax11_variables[(k, p)] + d11_variables[p] - 1

            constr_dict[f"new - axdd_30_k,p-ax{k, p}"] = (
                axdd30_variables[(k, p)],
                ax11_variables[(k, p)],
            )
            constr_dict[f"new - axdd_30_k,p-d00{k, p}"] = (
                axdd30_variables[(k, p)],
                d00_variables[p],
            )
            constr_dict[f"new - axdd_30_k,p{k, p}"] = (
                constr_30,
                axdd30_variables[(k, p)],
            )

            constr_dict[f"new - axdd_31_k,p-ax{k, p}"] = (
                axdd31_variables[(k, p)],
                ax11_variables[(k, p)],
            )
            constr_dict[f"new - axdd_31_k,p-d01{k, p}"] = (
                axdd31_variables[(k, p)],
                d01_variables[p],
            )
            constr_dict[f"new - axdd_31_k,p{k, p}"] = (
                constr_31,
                axdd31_variables[(k, p)],
            )

            constr_dict[f"new - axdd_32_k,p-ax{k, p}"] = (
                axdd32_variables[(k, p)],
                ax11_variables[(k, p)],
            )
            constr_dict[f"new - axdd_32_k,p-d10{k, p}"] = (
                axdd32_variables[(k, p)],
                d10_variables[p],
            )
            constr_dict[f"new - axdd_32_k,p{k, p}"] = (
                constr_32,
                axdd32_variables[(k, p)],
            )

            constr_dict[f"new - axdd_33_k,p-ax{k, p}"] = (
                axdd33_variables[(k, p)],
                ax11_variables[(k, p)],
            )
            constr_dict[f"new - axdd_33_k,p-d11{k, p}"] = (
                axdd33_variables[(k, p)],
                d11_variables[p],
            )
            constr_dict[f"new - axdd_33_k,p{k, p}"] = (
                constr_33,
                axdd33_variables[(k, p)],
            )

            if build:
                self.__model.add_constraint(
                    ax11_variables[(k, p)] >= axdd30_variables[(k, p)]
                )  # 30
                self.__model.add_constraint(
                    d00_variables[p] >= axdd30_variables[(k, p)]
                )  # 30
                self.__model.add_constraint(axdd30_variables[(k, p)] >= constr_30)  # 30

                self.__model.add_constraint(
                    ax11_variables[(k, p)] >= axdd31_variables[(k, p)]
                )  # 31
                self.__model.add_constraint(
                    d01_variables[p] >= axdd31_variables[(k, p)]
                )  # 31
                self.__model.add_constraint(axdd31_variables[(k, p)] >= constr_31)  # 31

                self.__model.add_constraint(
                    ax11_variables[(k, p)] >= axdd32_variables[(k, p)]
                )  # 32
                self.__model.add_constraint(
                    d10_variables[p] >= axdd32_variables[(k, p)]
                )  # 32
                self.__model.add_constraint(axdd32_variables[(k, p)] >= constr_32)  # 32

                self.__model.add_constraint(
                    ax11_variables[(k, p)] >= axdd33_variables[(k, p)]
                )  # 33
                self.__model.add_constraint(
                    d11_variables[p] >= axdd33_variables[(k, p)]
                )  # 33
                self.__model.add_constraint(axdd33_variables[(k, p)] >= constr_33)  # 33

        if not build:
            return constr_dict, equal_constr_dict

    def ax_dd_slack_variables_old(
        self,
        d00_variables,
        d01_variables,
        d10_variables,
        d11_variables,
        ax00_variables,
        ax01_variables,
        ax10_variables,
        ax11_variables,
        axdd00_variables,
        axdd01_variables,
        axdd02_variables,
        axdd03_variables,
        axdd10_variables,
        axdd11_variables,
        axdd12_variables,
        axdd13_variables,
        axdd20_variables,
        axdd21_variables,
        axdd22_variables,
        axdd23_variables,
        axdd30_variables,
        axdd31_variables,
        axdd32_variables,
        axdd33_variables,
        build=True,
    ):
        """
        Define xsp slack variables

        The slack variable:
        axdd00[k,p,q]==1 if ax00[k,p]==1 and dd00[p]==1
        axdd01[k,p,q]==1 if ax00[k,p]==1 and dd01[p]==1
        axdd02[k,p,q]==1 if ax00[k,p]==1 and dd10[p]==1
        axdd03[k,p,q]==1 if ax00[k,p]==1 and dd11[p]==1

        axdd10[k,p,q]==1 if ax01[k,p]==1 and dd00[p]==1
        axdd11[k,p,q]==1 if ax01[k,p]==1 and dd01[p]==1
        axdd12[k,p,q]==1 if ax01[k,p]==1 and dd00[p]==1
        axdd13[k,p,q]==1 if ax01[k,p]==1 and dd01[p]==1

        axdd20[k,p,q]==1 if ax10[k,p]==1 and dd00[p]==1
        axdd21[k,p,q]==1 if ax10[k,p]==1 and dd01[p]==1
        axdd22[k,p,q]==1 if ax10[k,p]==1 and dd00[p]==1
        axdd23[k,p,q]==1 if ax10[k,p]==1 and dd01[p]==1

        axdd30[k,p,q]==1 if ax11[k,p]==1 and dd00[p]==1
        axdd31[k,p,q]==1 if ax11[k,p]==1 and dd01[p]==1
        axdd32[k,p,q]==1 if ax11[k,p]==1 and dd10[p]==1
        axdd33[k,p,q]==1 if ax11[k,p]==1 and dd11[p]==1

        Note: Due to the constraint on the ax slack variables (and) dd slack variables, one of all dd and ax
        (00, 01, 10, 11) slack variables always has to be equal to one. Thereby not all combinations of possible 0 or 1
        values must be checked.

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        constr_dict = {}
        equal_constr_dict = {}
        for k, p in self.__allowed_variables["axdd"]:
            constr_00 = ax00_variables[(k, p)] + d00_variables[p]
            constr_01 = ax00_variables[(k, p)] + d01_variables[p]
            constr_02 = ax00_variables[(k, p)] + d10_variables[p]
            constr_03 = ax00_variables[(k, p)] + d11_variables[p]
            constr_10 = ax01_variables[(k, p)] + d00_variables[p]
            constr_11 = ax01_variables[(k, p)] + d01_variables[p]
            constr_12 = ax01_variables[(k, p)] + d10_variables[p]
            constr_13 = ax01_variables[(k, p)] + d11_variables[p]
            constr_20 = ax10_variables[(k, p)] + d00_variables[p]
            constr_21 = ax10_variables[(k, p)] + d01_variables[p]
            constr_22 = ax10_variables[(k, p)] + d10_variables[p]
            constr_23 = ax10_variables[(k, p)] + d11_variables[p]
            constr_30 = ax11_variables[(k, p)] + d00_variables[p]
            constr_31 = ax11_variables[(k, p)] + d01_variables[p]
            constr_32 = ax11_variables[(k, p)] + d10_variables[p]
            constr_33 = ax11_variables[(k, p)] + d11_variables[p]

            constr_dict[f"axdd_00_k,p{k, p}"] = (
                2 * axdd00_variables[(k, p)],
                constr_00,
            )
            constr_dict[f"axdd_01_k,p{k, p}"] = (
                2 * axdd01_variables[(k, p)],
                constr_01,
            )
            constr_dict[f"axdd_02_k,p{k, p}"] = (
                2 * axdd02_variables[(k, p)],
                constr_02,
            )
            constr_dict[f"axdd_03_k,p{k, p}"] = (
                2 * axdd03_variables[(k, p)],
                constr_03,
            )
            constr_dict[f"axdd_10_k,p{k, p}"] = (
                2 * axdd10_variables[(k, p)],
                constr_10,
            )
            constr_dict[f"axdd_11_k,p{k, p}"] = (
                2 * axdd11_variables[(k, p)],
                constr_11,
            )
            constr_dict[f"axdd_12_k,p{k, p}"] = (
                2 * axdd12_variables[(k, p)],
                constr_12,
            )
            constr_dict[f"axdd_13_k,p{k, p}"] = (
                2 * axdd13_variables[(k, p)],
                constr_13,
            )
            constr_dict[f"axdd_20_k,p{k, p}"] = (
                2 * axdd20_variables[(k, p)],
                constr_20,
            )
            constr_dict[f"axdd_21_k,p{k, p}"] = (
                2 * axdd21_variables[(k, p)],
                constr_21,
            )
            constr_dict[f"axdd_22_k,p{k, p}"] = (
                2 * axdd22_variables[(k, p)],
                constr_22,
            )
            constr_dict[f"axdd_23_k,p{k, p}"] = (
                2 * axdd23_variables[(k, p)],
                constr_23,
            )
            constr_dict[f"axdd_30_k,p{k, p}"] = (
                2 * axdd30_variables[(k, p)],
                constr_30,
            )
            constr_dict[f"axdd_31_k,p{k, p}"] = (
                2 * axdd31_variables[(k, p)],
                constr_31,
            )
            constr_dict[f"axdd_32_k,p{k, p}"] = (
                2 * axdd32_variables[(k, p)],
                constr_32,
            )
            constr_dict[f"axdd_33_k,p{k, p}"] = (
                2 * axdd33_variables[(k, p)],
                constr_33,
            )

            total_axdd = (
                axdd00_variables[(k, p)]
                + axdd01_variables[(k, p)]
                + axdd02_variables[(k, p)]
                + axdd03_variables[(k, p)]
                + axdd10_variables[(k, p)]
                + axdd11_variables[(k, p)]
                + axdd12_variables[(k, p)]
                + axdd13_variables[(k, p)]
                + axdd20_variables[(k, p)]
                + axdd21_variables[(k, p)]
                + axdd22_variables[(k, p)]
                + axdd23_variables[(k, p)]
                + axdd30_variables[(k, p)]
                + axdd31_variables[(k, p)]
                + axdd32_variables[(k, p)]
                + axdd33_variables[(k, p)]
            )

            equal_constr_dict[f"axdd=1_(k,p)={(k, p)}"] = (total_axdd, 1)
            if build:
                self.__model.add_constraint(
                    constr_00 >= 2 * axdd00_variables[(k, p)]
                )  # 00
                self.__model.add_constraint(
                    constr_01 >= 2 * axdd01_variables[(k, p)]
                )  # 01
                self.__model.add_constraint(
                    constr_02 >= 2 * axdd02_variables[(k, p)]
                )  # 02
                self.__model.add_constraint(
                    constr_03 >= 2 * axdd03_variables[(k, p)]
                )  # 03

                self.__model.add_constraint(
                    constr_10 >= 2 * axdd10_variables[(k, p)]
                )  # 10
                self.__model.add_constraint(
                    constr_11 >= 2 * axdd11_variables[(k, p)]
                )  # 11
                self.__model.add_constraint(
                    constr_12 >= 2 * axdd12_variables[(k, p)]
                )  # 12
                self.__model.add_constraint(
                    constr_13 >= 2 * axdd13_variables[(k, p)]
                )  # 13

                self.__model.add_constraint(
                    constr_20 >= 2 * axdd20_variables[(k, p)]
                )  # 20
                self.__model.add_constraint(
                    constr_21 >= 2 * axdd21_variables[(k, p)]
                )  # 21
                self.__model.add_constraint(
                    constr_22 >= 2 * axdd22_variables[(k, p)]
                )  # 22
                self.__model.add_constraint(
                    constr_23 >= 2 * axdd23_variables[(k, p)]
                )  # 23

                self.__model.add_constraint(
                    constr_30 >= 2 * axdd30_variables[(k, p)]
                )  # 30
                self.__model.add_constraint(
                    constr_31 >= 2 * axdd31_variables[(k, p)]
                )  # 31
                self.__model.add_constraint(
                    constr_32 >= 2 * axdd32_variables[(k, p)]
                )  # 32
                self.__model.add_constraint(
                    constr_33 >= 2 * axdd33_variables[(k, p)]
                )  # 33

                self.__model.add_constraint(total_axdd == 1)

        if not build:
            return constr_dict, equal_constr_dict

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
        """Each platform has maximally one car assigned to it"""
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
        axdd30_variables,
        axdd31_variables,
        axdd32_variables,
        axdd33_variables,
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
                    * (
                        self.data.lr_coefficient[self.data.vehicles[k]["Class"]][0][0]
                        * axdd30_variables[(k, self.data.Pa.index(p))]
                        + self.data.lr_coefficient[self.data.vehicles[k]["Class"]][0][1]
                        * axdd31_variables[(k, self.data.Pa.index(p))]
                        + self.data.lr_coefficient[self.data.vehicles[k]["Class"]][1][0]
                        * axdd32_variables[(k, self.data.Pa.index(p))]
                        + self.data.lr_coefficient[self.data.vehicles[k]["Class"]][1][1]
                        * axdd33_variables[(k, self.data.Pa.index(p))]
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
                # print("lin build angled_L", angled_L)
                # print("build not_angled_L", not_angled_L)
                # print("build max_constr", max_constr)
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

    def length_constraints_no_d(
        self,
        x_variables,
        a_variables,
        sp_variables,
        ax11_variables,
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
                    * ax11_variables[(k, self.data.Pa.index(p))]
                    * self.data.lr_coefficient_no_d[self.data.vehicles[k]["Class"]]
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
                # print("lin build angled_L", angled_L)
                # print("build not_angled_L", not_angled_L)
                # print("build max_constr", max_constr)
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
        axdd30_variables,
        axdd31_variables,
        axdd32_variables,
        axdd33_variables,
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
                    * (
                        self.data.hr_coefficient[self.data.vehicles[k]["Class"]][0][0]
                        * axdd30_variables[(k, self.data.Pa.index(p))]
                        + self.data.hr_coefficient[self.data.vehicles[k]["Class"]][0][1]
                        * axdd31_variables[(k, self.data.Pa.index(p))]
                        + self.data.hr_coefficient[self.data.vehicles[k]["Class"]][1][0]
                        * axdd32_variables[(k, self.data.Pa.index(p))]
                        + self.data.hr_coefficient[self.data.vehicles[k]["Class"]][1][1]
                        * axdd33_variables[(k, self.data.Pa.index(p))]
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

    def height_constraint_no_d(self, x_variables, ax11_variables, build=True):
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
                    * ax11_variables[(k, self.data.Pa.index(p))]
                    * self.data.hr_coefficient_no_d[self.data.vehicles[k]["Class"]]
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
        self,
        x_variables,
        a_variables,
        ax11_variables,
        xsp11_variables,
        axsp31_variables,
        build=True,
    ):
        """
        Weight constraints

        Returns: the constraint dictionary if build=False and sets the constraint for self.__model if build=True
        """
        constr_dict = {}
        """Platforms than can not be angled, have to be checkd so that the weight
        of their assigned car does not exceed (gamma:) the weight limit of either
        the platform or the weight limit of a combined configuration with another
        platform"""
        for p in set(self.data.P).difference(self.data.Pa_set):
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
                        xsp11_variables[(k, p, q_idx)]
                        * (self.data.wc_max[q_idx] - self.data.wp_max[p])
                        + x_variables[(k, p)] * self.data.wp_max[p]
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
                        xsp11_variables[(k, p, q_idx)]
                        * (self.data.wc_max[q_idx] - self.data.wp_max[p])
                        + x_variables[(k, p)] * self.data.wp_max[p]
                        for k in range(self.data.K)
                    ]
                )
                angle_constr = (
                    a_variables[self.data.Pa.index(p)]
                    * self.data.wa_max[self.data.Pa.index(p)]
                )

                alpha_gamma_constr = sum(
                    [
                        axsp31_variables[(k, p, q_idx)]
                        * (self.data.wc_max[q_idx] - self.data.wp_max[p])
                        + ax11_variables[(k, self.data.Pa.index(p))]
                        * self.data.wp_max[p]
                        for k in range(self.data.K)
                    ]
                )

                max_sp_a_weight = -alpha_gamma_constr + angle_constr + gamma_constraint
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

        xsp_liste = []
        for q in range(len(self.data.Psp)):
            for p in self.data.Psp[q]:
                for k in range(self.data.K):
                    xsp_liste.append((k, p, q))

        axsp_liste1 = []
        for q in range(len(self.data.Psp)):
            for _p in self.data.Psp[q]:
                if _p in self.data.Pa:
                    for k in range(self.data.K):
                        axsp_liste1.append((k, _p, q))

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
            "xsp": xsp_liste,
            "axsp": axsp_liste1,
            "axdd": [
                (k, p)
                for k, p in itertools.product(
                    range(self.data.K), range(len(self.data.Pa))
                )
            ],
        }

    @property
    def variables(self):
        return self.__variables

    @property
    def model(self):
        return self.__model

    def check_solution_test(self, solution):
        """
        all constraints given back are of the format (constr, max_constr).
        If max_constr - constr > 0 it is fulfilled and if <0 not.

        While the tupel values of the equal constraints have to be equal.
        """

        all_constraints = {}
        equal_constr = {}

        if self.d_var_exists and not self.old_sack_var:
            x_variables = solution["x_variables"]
            a_variables = solution["a_variables"]
            sp_variables = solution["sp_variables"]

            d_variables = solution["d_variables"]

            d00_variables = solution["d00_variables"]
            d01_variables = solution["d01_variables"]
            d10_variables = solution["d10_variables"]
            d11_variables = solution["d11_variables"]

            axdd30_variables = solution["axdd30_variables"]
            axdd31_variables = solution["axdd31_variables"]
            axdd32_variables = solution["axdd32_variables"]
            axdd33_variables = solution["axdd33_variables"]

            axsp31_variables = solution["axsp31_variables"]

            ax11_variables = solution["ax11_variables"]

            xsp11_variables = solution["xsp11_variables"]

            dd_constraints, dd_equal_constr = self.dd_slack_variables(
                d_variable=d_variables,
                d00_variables=d00_variables,
                d01_variables=d01_variables,
                d10_variables=d10_variables,
                d11_variables=d11_variables,
                build=False,
            )
            all_constraints.update(dd_constraints)
            equal_constr.update(dd_equal_constr)

            axdd_constraints, axdd_equal_constr = self.ax_dd_slack_variables(
                d00_variables=d00_variables,
                d01_variables=d01_variables,
                d10_variables=d10_variables,
                d11_variables=d11_variables,
                ax11_variables=ax11_variables,
                axdd30_variables=axdd30_variables,
                axdd31_variables=axdd31_variables,
                axdd32_variables=axdd32_variables,
                axdd33_variables=axdd33_variables,
                build=False,
            )
            all_constraints.update(axdd_constraints)
            equal_constr.update(axdd_equal_constr)

            ax_constraints, ax_equal_constr = self.ax_slack_variables(
                a_variables=a_variables,
                x_variables=x_variables,
                ax11_variables=ax11_variables,
                build=False,
            )
            all_constraints.update(ax_constraints)
            equal_constr.update(ax_equal_constr)

            xsp_constraints, xsp_equal_constr = self.xsp_slack_variable(
                x_variables=x_variables,
                sp_variables=sp_variables,
                xsp11_variables=xsp11_variables,
                build=False,
            )
            all_constraints.update(xsp_constraints)
            equal_constr.update(xsp_equal_constr)

            axsp_constraints, axsp_equal_constr = self.axsp_slack_variables(
                sp_variables=sp_variables,
                ax11_variables=ax11_variables,
                axsp31_variables=axsp31_variables,
                build=False,
            )
            all_constraints.update(axsp_constraints)
            equal_constr.update(axsp_equal_constr)

            length_constraints = self.length_constraints(
                x_variables=x_variables,
                a_variables=a_variables,
                sp_variables=sp_variables,
                axdd30_variables=axdd30_variables,
                axdd31_variables=axdd31_variables,
                axdd32_variables=axdd32_variables,
                axdd33_variables=axdd33_variables,
                build=False,
            )
            all_constraints.update(length_constraints)

            height_constraints = self.height_constraint(
                x_variables=x_variables,
                axdd30_variables=axdd30_variables,
                axdd31_variables=axdd31_variables,
                axdd32_variables=axdd32_variables,
                axdd33_variables=axdd33_variables,
                build=False,
            )
            all_constraints.update(height_constraints)

        elif self.d_var_exists and self.old_sack_var:
            x_variables = solution["x_variables"]
            a_variables = solution["a_variables"]
            sp_variables = solution["sp_variables"]

            d_variables = solution["d_variables"]

            d00_variables = solution["d00_variables"]
            d01_variables = solution["d01_variables"]
            d10_variables = solution["d10_variables"]
            d11_variables = solution["d11_variables"]

            ax00_variables = solution["ax00_variables"]
            ax01_variables = solution["ax01_variables"]
            ax10_variables = solution["ax10_variables"]
            ax11_variables = solution["ax11_variables"]

            xsp00_variables = solution["xsp00_variables"]
            xsp01_variables = solution["xsp01_variables"]
            xsp10_variables = solution["xsp10_variables"]
            xsp11_variables = solution["xsp11_variables"]

            axsp00_variables = solution["axsp00_variables"]
            axsp01_variables = solution["axsp01_variables"]
            axsp10_variables = solution["axsp10_variables"]
            axsp11_variables = solution["axsp11_variables"]
            axsp20_variables = solution["axsp20_variables"]
            axsp21_variables = solution["axsp21_variables"]
            axsp30_variables = solution["axsp30_variables"]
            axsp31_variables = solution["axsp31_variables"]

            axdd00_variables = solution["axdd00_variables"]
            axdd01_variables = solution["axdd01_variables"]
            axdd02_variables = solution["axdd02_variables"]
            axdd03_variables = solution["axdd03_variables"]
            axdd10_variables = solution["axdd10_variables"]
            axdd11_variables = solution["axdd11_variables"]
            axdd12_variables = solution["axdd12_variables"]
            axdd13_variables = solution["axdd13_variables"]
            axdd20_variables = solution["axdd20_variables"]
            axdd21_variables = solution["axdd21_variables"]
            axdd22_variables = solution["axdd22_variables"]
            axdd23_variables = solution["axdd23_variables"]
            axdd30_variables = solution["axdd30_variables"]
            axdd31_variables = solution["axdd31_variables"]
            axdd32_variables = solution["axdd32_variables"]
            axdd33_variables = solution["axdd33_variables"]

            dd_constraints, dd_equal_constr = self.dd_slack_variables(
                d_variable=d_variables,
                d00_variables=d00_variables,
                d01_variables=d01_variables,
                d10_variables=d10_variables,
                d11_variables=d11_variables,
                build=False,
            )
            all_constraints.update(dd_constraints)
            equal_constr.update(dd_equal_constr)

            axdd_constraints, axdd_equal_constr = self.ax_dd_slack_variables_old(
                d00_variables=d00_variables,
                d01_variables=d01_variables,
                d10_variables=d10_variables,
                d11_variables=d11_variables,
                ax00_variables=ax00_variables,
                ax01_variables=ax01_variables,
                ax10_variables=ax10_variables,
                ax11_variables=ax11_variables,
                axdd00_variables=axdd00_variables,
                axdd01_variables=axdd01_variables,
                axdd02_variables=axdd02_variables,
                axdd03_variables=axdd03_variables,
                axdd10_variables=axdd10_variables,
                axdd11_variables=axdd11_variables,
                axdd12_variables=axdd12_variables,
                axdd13_variables=axdd13_variables,
                axdd20_variables=axdd20_variables,
                axdd21_variables=axdd21_variables,
                axdd22_variables=axdd22_variables,
                axdd23_variables=axdd23_variables,
                axdd30_variables=axdd30_variables,
                axdd31_variables=axdd31_variables,
                axdd32_variables=axdd32_variables,
                axdd33_variables=axdd33_variables,
                build=False,
            )
            all_constraints.update(axdd_constraints)
            equal_constr.update(axdd_equal_constr)

            ax_constraints, ax_equal_constr = self.ax_slack_variables_old(
                a_variables=a_variables,
                x_variables=x_variables,
                ax00_variables=ax00_variables,
                ax01_variables=ax01_variables,
                ax10_variables=ax10_variables,
                ax11_variables=ax11_variables,
                build=False,
            )
            all_constraints.update(ax_constraints)
            equal_constr.update(ax_equal_constr)

            xsp_constraints, xsp_equal_constr = self.xsp_slack_variable_old(
                x_variables=x_variables,
                sp_variables=sp_variables,
                xsp00_variables=xsp00_variables,
                xsp01_variables=xsp01_variables,
                xsp10_variables=xsp10_variables,
                xsp11_variables=xsp11_variables,
                build=False,
            )
            all_constraints.update(xsp_constraints)
            equal_constr.update(xsp_equal_constr)

            axsp_constraints, axsp_equal_constr = self.axsp_slack_variables_old(
                sp_variables=sp_variables,
                ax00_variables=ax00_variables,
                ax01_variables=ax01_variables,
                ax10_variables=ax10_variables,
                ax11_variables=ax11_variables,
                axsp00_variables=axsp00_variables,
                axsp01_variables=axsp01_variables,
                axsp10_variables=axsp10_variables,
                axsp11_variables=axsp11_variables,
                axsp20_variables=axsp20_variables,
                axsp21_variables=axsp21_variables,
                axsp30_variables=axsp30_variables,
                axsp31_variables=axsp31_variables,
                build=False,
            )
            all_constraints.update(axsp_constraints)
            equal_constr.update(axsp_equal_constr)

            # length constraints
            length_constraints = self.length_constraints(
                x_variables=x_variables,
                a_variables=a_variables,
                sp_variables=sp_variables,
                axdd30_variables=axdd30_variables,
                axdd31_variables=axdd31_variables,
                axdd32_variables=axdd32_variables,
                axdd33_variables=axdd33_variables,
                build=False,
            )
            all_constraints.update(length_constraints)

            height_constraints = self.height_constraint(
                x_variables=x_variables,
                axdd30_variables=axdd30_variables,
                axdd31_variables=axdd31_variables,
                axdd32_variables=axdd32_variables,
                axdd33_variables=axdd33_variables,
                build=False,
            )
            all_constraints.update(height_constraints)

        elif not self.d_var_exists and self.old_sack_var:
            x_variables = solution["x_variables"]
            a_variables = solution["a_variables"]
            sp_variables = solution["sp_variables"]

            ax00_variables = solution["ax00_variables"]
            ax01_variables = solution["ax01_variables"]
            ax10_variables = solution["ax10_variables"]
            ax11_variables = solution["ax11_variables"]

            xsp00_variables = solution["xsp00_variables"]
            xsp01_variables = solution["xsp01_variables"]
            xsp10_variables = solution["xsp10_variables"]
            xsp11_variables = solution["xsp11_variables"]

            axsp00_variables = solution["axsp00_variables"]
            axsp01_variables = solution["axsp01_variables"]
            axsp10_variables = solution["axsp10_variables"]
            axsp11_variables = solution["axsp11_variables"]
            axsp20_variables = solution["axsp20_variables"]
            axsp21_variables = solution["axsp21_variables"]
            axsp30_variables = solution["axsp30_variables"]
            axsp31_variables = solution["axsp31_variables"]

            ax_constraints, ax_equal_constr = self.ax_slack_variables_old(
                a_variables=a_variables,
                x_variables=x_variables,
                ax00_variables=ax00_variables,
                ax01_variables=ax01_variables,
                ax10_variables=ax10_variables,
                ax11_variables=ax11_variables,
                build=False,
            )
            all_constraints.update(ax_constraints)
            equal_constr.update(ax_equal_constr)

            xsp_constraints, xsp_equal_constr = self.xsp_slack_variable_old(
                x_variables=x_variables,
                sp_variables=sp_variables,
                xsp00_variables=xsp00_variables,
                xsp01_variables=xsp01_variables,
                xsp10_variables=xsp10_variables,
                xsp11_variables=xsp11_variables,
                build=False,
            )
            all_constraints.update(xsp_constraints)
            equal_constr.update(xsp_equal_constr)

            axsp_constraints, axsp_equal_constr = self.axsp_slack_variables_old(
                sp_variables=sp_variables,
                ax00_variables=ax00_variables,
                ax01_variables=ax01_variables,
                ax10_variables=ax10_variables,
                ax11_variables=ax11_variables,
                axsp00_variables=axsp00_variables,
                axsp01_variables=axsp01_variables,
                axsp10_variables=axsp10_variables,
                axsp11_variables=axsp11_variables,
                axsp20_variables=axsp20_variables,
                axsp21_variables=axsp21_variables,
                axsp30_variables=axsp30_variables,
                axsp31_variables=axsp31_variables,
                build=False,
            )
            all_constraints.update(axsp_constraints)
            equal_constr.update(axsp_equal_constr)

            # length constraints
            length_constraints = self.length_constraints_no_d(
                x_variables=x_variables,
                a_variables=a_variables,
                sp_variables=sp_variables,
                ax11_variables=ax11_variables,
                build=False,
            )
            all_constraints.update(length_constraints)

            # height constraints
            height_constraints = self.height_constraint_no_d(
                x_variables=x_variables, ax11_variables=ax11_variables, build=False
            )
            all_constraints.update(height_constraints)

        elif not self.d_var_exists and not self.old_sack_var:
            x_variables = solution["x_variables"]
            a_variables = solution["a_variables"]
            sp_variables = solution["sp_variables"]

            ax11_variables = solution["ax11_variables"]

            xsp11_variables = solution["xsp11_variables"]

            axsp31_variables = solution["axsp31_variables"]

            ax_constraints, ax_equal_constr = self.ax_slack_variables(
                a_variables=a_variables,
                x_variables=x_variables,
                ax11_variables=ax11_variables,
            )
            all_constraints.update(ax_constraints)
            equal_constr.update(ax_equal_constr)

            xsp_constraints, xsp_equal_constr = self.xsp_slack_variable(
                x_variables=x_variables,
                sp_variables=sp_variables,
                xsp11_variables=xsp11_variables,
            )
            all_constraints.update(xsp_constraints)
            equal_constr.update(xsp_equal_constr)

            axsp_constraints, axsp_equal_constr = self.axsp_slack_variables(
                sp_variables=sp_variables,
                ax11_variables=ax11_variables,
                axsp31_variables=axsp31_variables,
            )
            all_constraints.update(axsp_constraints)
            equal_constr.update(axsp_equal_constr)

            # length constraints
            length_constraints = self.length_constraints_no_d(
                x_variables=x_variables,
                a_variables=a_variables,
                sp_variables=sp_variables,
                ax11_variables=ax11_variables,
            )
            all_constraints.update(length_constraints)

            # height constraints
            height_constraints = self.height_constraint_no_d(
                x_variables=x_variables, ax11_variables=ax11_variables
            )
            all_constraints.update(height_constraints)

        obj_func = self.objective_function(variable=x_variables, build=False)

        # assignement constraints
        assign_constraints = self.assignment_constraints(
            x_variables=x_variables, sp_variables=sp_variables, build=False
        )
        all_constraints.update(assign_constraints)

        # weight constraints
        weight_constraints = self.weight_constraints(
            x_variables=x_variables,
            a_variables=a_variables,
            ax11_variables=ax11_variables,
            xsp11_variables=xsp11_variables,
            axsp31_variables=axsp31_variables,
            build=False,
        )
        all_constraints.update(weight_constraints)

        assign_constraints = self.assignment_constraints(
            x_variables=x_variables, sp_variables=sp_variables, build=False
        )
        all_constraints.update(assign_constraints)

        passed_count = 0
        failed_count = 0

        for name, values in equal_constr.items():
            Delta = values[1] - values[0]
            # print(name, values[0], values[1])
            if Delta == 0:
                # print("PASSED")
                passed_count += 1
            else:
                print(f"{name} - Failed", Delta)
                failed_count += 1

        for name, values in all_constraints.items():
            Delta = values[1] - values[0]
            # print(name, values[0], values[1])
            if Delta >= 0:
                # print("PASSED")
                passed_count += 1
            else:
                print(f"{name} - Failed", Delta)
                failed_count += 1

        print(f"Passed {passed_count} constraints")
        print(f"Failed {failed_count} constraints")
