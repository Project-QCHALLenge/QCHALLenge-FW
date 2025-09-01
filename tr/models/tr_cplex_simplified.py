from docplex.mp.model import Model
from tr.data.railnetwork import RailNetwork, Train
import networkx as nx
import numpy as np
from copy import deepcopy
from transformations.from_cplex import FromCPLEX
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import SampleSet
from abstract.models.abstract_model import AbstractModel


class TR_cplex_simple(AbstractModel):

    def __init__(self, railnetwork: RailNetwork, dmax: int = 2, tau_stop: int = 0):

        self.constraints = self.Constraints(self)
        self._solution = None
        self.railnetwork = railnetwork
        self.trains = railnetwork._trains  # dict {train.id : Train}
        if len(self.trains) == 0:
            raise ValueError("There are no trains in the network.")
        self._d_max = dmax  # TODO this should be determined by a heuristic

        self._model = Model(name="TrainScheduling")

        self._variables = []  # is this needed afterall?

        self.S = {
            j: train.route for j, train in self.trains.items()
        }  # station dict = {trainID : route_of_trainID}

        self.C = {
            j: self._get_subsequents(train.route) for j, train in self.trains.items()
        }  # dict {j:pairs of subsequent stations on route of j}

        self.tau_pass = {
            (j, s, s_next): round(
                self._edge_attr(s, s_next)[0]
                / min(train.speed, self._edge_attr(s, s_next)[1])
            )
            for j, train in self.trains.items()
            for (s, s_next) in self.C[j]
        }
        # {(j,"A","B"):time j needs to go from A to B (or back)}

        self.tau_blocks = np.zeros(
            (len(self.trains), len(self.railnetwork.network.nodes)), dtype=int
        )  # since we have no blocks at the moment, we don't have a safety distance
        self.tau_stop = np.full(
            (len(self.trains), len(self.railnetwork.network.nodes)), tau_stop
        )

        # for simplicity we assume all trains has the same minimum stopping time
        self.tau_res = np.zeros(
            (len(self.trains), len(self.railnetwork.network.nodes)), dtype=int
        )  # we also assume the switches and resources of the station can be accessed with no time

        self._schedule, self._max_time = self._create_schedule()
        self.upsilon = deepcopy(self._schedule)

        # {schedule dict that has the minimum departures without any additional delay
        self.w = {
            j: train.priority for j, train in self.trains.items()
        }  # dict train priorities

        self.sigma = deepcopy(self.upsilon)
        # planned schedule, only neccesary to make sure trains don't depart too early
        # TODO it should be the other way round, conflict free schedule -> add unavoidable delay -> conflicted schedule with no additional delay

        self.J_d = {
            (j, j_1): self._common_subsequents(j, j_1)
            for j in self.trains.keys()
            for j_1 in self.trains.keys()
            if j < j_1 and len(self._common_subsequents(j, j_1)) > 0
        }

        # pairs of subsequent stations j and j_1 have in common
        # {(1,2):(('C','D'), ('D','F'))}

        self.J_o = {
            (j, j_1): self._common_opposites(j, j_1)
            for j in self.trains.keys()
            for j_1 in self.trains.keys()
            if j < j_1 and len(self._common_opposites(j, j_1)) > 0
        }  # pairs of trains in opposite direction competing for a single track line

        self.t = {
            (j, s, i_o): self._model.integer_var(
                name=f"t_{j}_{s}_{i_o}",
                lb=0,  # self.upsilon[j, s, i_o] would be better, but dimod does only allow for interger vars with lb 0
                ub=self.upsilon[j, s, i_o] + self._d_max,
            )
            for j, train in self.trains.items()
            for s in self.S[j]
            for i_o in ["in", "out"]
        }  # t in minutes ranges from 0 up until the longest allowed delay + the last departure = max_time
        # to save variables one could also try to incorperate the no early departure constraint here
        # j:int -> train id in the network
        # s: str -> node name in the train network "A", "B", etc. on route of j
        # i_o: str ["in", "out"] arrival and departure

        self.y = {
            (j, j_1, tup[0]): self._model.binary_var(name=f"y_{j}_{j_1}_{tup[0]}")
            for (j, j_1), common_subsequents in self.J_d.items()
            for tup in common_subsequents
        }  # trains in J_d go to the same direction
        # y(j,j_1,s) = 1 => j leaves s before j_1

        for j, j_1, s in self.y.keys():
            self._model.add_constraint(
                self.y[(j, j_1, s)] == 1 - self.y[(j_1, j, s)],
                ctname=f"order_{j}_{j_1}_{s}",
            )

        self.z = {
            (j, j_1, tup): self._model.binary_var(name=f"z_{j}_{j_1}_{tup[0]}_{tup[1]}")
            for (j, j_1), common_opposites in self.J_o.items()
            for tup in common_opposites
        }  # trains in J_single_o go in opposite direction and compete for a single track between s and s_next
        # with j currently at s and j_1 at s_next
        # z(j,j_1,s,s_next) = 1 => j leaves s before j_1 leaves s_next

        for j, j_1, (s, s_next) in self.z.keys():
            self._model.add_constraint(
                self.z[(j, j_1, (s, s_next))] == 1 - self.z[(j_1, j, (s_next, s))],
                ctname=f"order_{j}_{j_1}_{s}_{s_next}",
            )

        self._build_model()

    def check_violation(self, variables: dict[str, float] = None):
        """Checks whether a given variables assignment violates contraints."""
        self.violations = dict()

        t, y, z = self._make_dicts(variables)

        for (j, j1, station), v in y.items():
            if y[j, j1, station] != 1 - y[(j1, j, station)]:
                if not ("y", j, j1, station) in self.violations:
                    self.violations["y", j1, j, station] = (
                        f"precedence variable violation"
                    )

        for (j, j1, s1, s2), v in z.items():
            if z[j, j1, s1, s2] != 1 - z[j1, j, s2, s1]:
                if not ("z", j, j1, s1, s2) in self.violations:
                    self.violations["z", j1, j, s2, s1] = (
                        f"precedence variable violation"
                    )

        for constraint in self.constraints.active_constraints:
            self._check_constraint(constraint, t, y, z)

        return self.violations

    def get_model(self):
        return self._model

    def set_dmax(self, dmax):
        self.__init__(
            self.railnetwork, dmax
        )  # or rebuild self.t which is probably the correct way to do it

    def set_bqm(self):
        self.bqm, self.inverter = FromCPLEX(self._model).to_bqm()

    def solve(self, solver="cplex", quiet=False, **params):
        if solver == "cplex":
            self._solution = self._model.solve(**params)
            vars = {
                var.name: self._solution.get_value(var)
                for var in self._model.iter_variables()
            }
            sol = vars.copy()
            for var, val in vars.items():
                if var[0] == "z":
                    s = var.split("_")
                    sol[f"z_{s[2]}_{s[1]}_{s[4]}_{s[3]}"] = 1 - val
                if var[0] == "y":
                    s = var.split("_")
                    sol[f"y_{s[2]}_{s[1]}_{s[3]}"] = 1 - val
            return sol

        elif solver == "SA":

            if self.sa_solver is None:
                self.sa_solver = SimulatedAnnealingSampler()
                sa_res = self.sa_solver.sample(bqm, **params)
                sa_sol = self._get_feasible(
                    sample_set=sa_res, inverter=inverter, quiet=quiet
                )

                return sa_sol

        elif solver == "DW":
            if self.dw_solver is None:
                ds = DWaveSampler()
                self.dw_solver = EmbeddingComposite(ds)
                if self.bqm is None:
                    bqm, inverter = FromCPLEX(self._model).to_bqm()
                else:
                    bqm = self.bqm
                    inverter = self.inverter
                dw_res = self.dw_solver.sample(bqm, **params)
                dw_sol = self._get_feasible(
                    sample_set=dw_res, inverter=inverter, quiet=quiet
                )
                return dw_sol

        elif solver == "gurobi":
            raise NotImplementedError("Not yet implemented.")
        else:
            raise ValueError("No vaild solver. (cplex, SA, DW, gurobi)")

    @property
    def solution(self, verbose=False):
        if verbose:
            print("Objective value: ", self._solution.get_objective_value())
        sol = {}
        for var in self._model.iter_variables():
            if verbose:
                print(var, ":", round(self._solution.get_value(var), 2))
            sol[var.name] = round(self._solution.get_value(var), 2)
        return self._solution.get_objective_value(), sol

    ### helpers

    def _get_feasible(self, sample_set: SampleSet, inverter, quiet: bool):
        for sample in sample_set.samples():
            vars = inverter(sample)
            violations = self.check_violation(vars)
            if len(violations) == 0:
                return vars
        if not quiet:
            print("no feasible solution found, returning best unfeasible solution")
        return inverter(sample_set.first.sample)

    def _check_constraint(self, constraint, t, y, z):

        if constraint == "min_passing":
            self._check_min_passing(t, y, z)
        elif constraint == "min_headway":
            self._check_min_headway(t, y, z)
        elif constraint == "single_track":
            self._check_single_track(t, y, z)
        elif constraint == "min_stop":
            self._check_min_stop(t, y, z)
        elif constraint == "no_early_departure":
            self._check_no_early_departure(t, y, z)
        else:
            raise ValueError("Constraint not found")

    def _create_schedule(self) -> tuple[dict, int]:
        status = ["in", "out"]
        schedule = {
            (j, s, status[pos]): list([t_in, t_out])[pos]
            for j, train in self.railnetwork._trains.items()
            for (s, t_in, t_out) in train.schedule
            for pos in [0, 1]
        }
        max_time = max(schedule.values())
        return schedule, max_time

    def _edge_attr(self, x, y):
        """
        returns distance and max speed of track between nodes x and y
        """
        edge_attributes = self.railnetwork.network.get_edge_data(x, y)

        distance = edge_attributes["distance"]
        max_speed = edge_attributes["max_speed"]
        return distance, max_speed

    def _get_subsequents(self, tup):
        """
        from a route ("A","B","C") create subsequent pairs-> (("A","B"),("B","C"))
        """
        pairs = list()
        for i in range(len(tup) - 1):
            pairs.append((tup[i], tup[i + 1]))
        return tuple(pairs)

    def _common_subsequents(self, j, j_1):
        tup1 = self.C[j]
        tup2 = self.C[j_1]
        pairs = list()
        for a, b in tup1:
            if (a, b) in tup2:
                if (
                    self._schedule[j, a, "out"] <= self._schedule[j_1, a, "out"]
                    and self._schedule[j_1, a, "out"]
                    < self._schedule[j, b, "in"] + self._d_max
                ) or (
                    self._schedule[j_1, a, "out"] <= self._schedule[j, a, "out"]
                    and self._schedule[j, a, "out"]
                    < self._schedule[j_1, b, "in"] + self._d_max
                ):
                    pairs.append((a, b))
        return tuple(pairs)

    def _common_opposites(self, j, j_1):
        tup1 = self.C[j]
        tup2 = self.C[j_1]
        pairs = list()
        for a, b in tup1:
            if (b, a) in tup2:
                if (
                    self._schedule[j, a, "out"] <= self._schedule[j_1, b, "out"]
                    and self._schedule[j_1, b, "out"]
                    < self._schedule[j, b, "in"] + self._d_max
                ) or (
                    self._schedule[j_1, b, "out"] <= self._schedule[j, a, "out"]
                    and self._schedule[j, a, "out"]
                    < self._schedule[j_1, a, "in"] + self._d_max
                ):
                    pairs.append((a, b))
        return tuple(pairs)

    def build_model(self, *args, **kwargs):
        self._build_model()

    def _build_model(self):

        # Objective function
        objective_expr = self._model.sum(
            self.w[j]
            * self._model.sum(
                self.t[j, s, "out"] - self.upsilon[j, s, "out"] for s in self.S[j]
            )
            for j, train in self.trains.items()
        )
        self._model.minimize(objective_expr)

        # variable ranges
        for j, train in self.trains.items():
            for s in self.S[j]:
                self._model.add_constraint(
                    self.t[j, s, "out"] >= self.upsilon[j, s, "out"]
                )
                self._model.add_constraint(
                    self.t[j, s, "out"] <= self.upsilon[j, s, "out"] + self._d_max
                )

    def _make_dicts(self, solution):
        t, y, z = {}, {}, {}
        for k, v in solution.items():
            s = k.split("_")
            if s[0] == "t":
                t[(int(s[1]), s[2], s[3])] = v
            elif s[0] == "y":
                y[(int(s[1]), int(s[2]), s[3])] = v
            elif s[0] == "z":
                z[(int(s[1]), int(s[2]), s[3], s[4])] = v
            else:
                raise ValueError(f"{k} is not a valid variable")
        return t, y, z

    def objectiv_from_solution(self, solution, only_feasible=True, quiet=False):
        if only_feasible:
            violations = self.check_violation(solution)
            if len(violations) > 0:
                print(
                    f"Solution not feasible. {len(violations)} constraint violation(s):"
                )
                return violations
        obj = 0

        t, _, _ = self._make_dicts(solution)
        for k, v in self.upsilon.items():
            obj += t[k] - v
        return v

    #### constraints
    def _min_passing(self):
        for j, train in self.trains.items():
            for s, s_next in self.C[j]:
                self._model.add_constraint(
                    self.t[j, s_next, "in"] - self.t[j, s, "out"]
                    == self.tau_pass[j, s, s_next],
                    ctname=f"min_passing_ilp_{j}_{s}_{s_next}",
                )

    def _check_min_passing(self, t, y, z):
        for j, train in self.trains.items():

            for s, s_next in self.C[j]:
                if t[j, s_next, "in"] - t[j, s, "out"] != self.tau_pass[j, s, s_next]:
                    self.violations[(j, s, s_next)] = "min_passing"

    def _min_headway(self):
        # min headway constraint
        M = 1000  # the big M to implement implications

        for (j, j_1), tups in self.J_d.items():
            for s, s_next in tups:
                self._model.add_constraint(
                    self.t[j_1, s, "out"]
                    + M * (1 - self.y[j, j_1, s])
                    - self.t[j, s, "out"]
                    >= self.tau_pass[j, s, s_next],
                    ctname=f"min_headway_ilp_{j}_{j_1}_{s}_{s_next}",
                )

    def _check_min_headway(self, t, y, z):
        M = 1000

        for (j, j_1), tups in self.J_d.items():
            for s, s_next in tups:
                condition = (
                    t[j_1, s, "out"] + M * (1 - y[j, j_1, s]) - t[j, s, "out"]
                    >= self.tau_pass[j, s, s_next]
                )
                if not condition:
                    if not (j_1, j, s) in self.violations:
                        self.violations[(j, j_1, s)] = "min_headway"

    def _single_track(self):
        # single track constraint
        M = 1000  # the big M to implement implications
        for (j, j_1), tups in self.J_o.items():
            for s, s_next in tups:
                self._model.add_constraint(
                    self.t[j_1, s_next, "out"] + M * (1 - self.z[j, j_1, (s, s_next)])
                    >= self.t[j, s_next, "in"],
                    ctname=f"single_track_ilp_{j}_{j_1}_{(s,s_next)}",
                )

    def _check_single_track(self, t, y, z):
        M = 1000  # the big M to implement implications
        for (j, j_1), tups in self.J_o.items():
            for s, s_next in tups:
                condition = (
                    t[j_1, s_next, "out"] + M * (1 - z[j, j_1, s, s_next])
                    >= t[j, s_next, "in"]
                )
            if not condition:
                if not (j_1, j, s_next, s) in self.violations:
                    self.violations[j, j_1, s, s_next] = "single_track"

    def _min_stop(self):
        for j, train in self.trains.items():
            for s in train.route:
                self._model.add_constraint(
                    self.t[j, s, "out"] - self.t[j, s, "in"] >= 1,  # hard coded TODO
                    ctname=f"min_stop_{j}_{s}",
                )

    def _check_min_stop(self, t, y, z):
        for j, train in self.trains.items():
            for s in train.route:
                if t[j, s, "out"] - t[j, s, "in"] < 1:  # hard coded TODO
                    self.violations[j, s] = "min_stop"

    # no early departure constraint
    def _no_early_departure(self):
        for j, train in self.trains.items():
            for s in self.S[j]:
                self._model.add_constraint(
                    self.t[j, s, "out"] >= self.sigma[j, s, "out"],
                    ctname=f"no_early_departure_{j}_{s}",
                )

    def _check_no_early_departure(self, t, y, z):
        for j, train in self.trains.items():
            for s in self.S[j]:
                if t[j, s, "out"] < self.sigma[j, s, "out"]:
                    self.violations[j, s] = "no_early_departure"

    # for easier adding of constrainsts
    class Constraints:
        def __init__(self, parent) -> None:
            self.parent = parent
            self.active_constraints = []

        def show(self):
            if len(self.active_constraints) > 0:
                print(self.active_constraints)

            else:
                print(None)

        def all(self):
            self.min_passing()
            self.min_headway()
            self.min_stop()
            self.single_track()
            self.no_early_departure()

        def min_passing(self):
            if "min_passing" in self.active_constraints:
                print("Contraint already active.")
            else:
                self.parent._min_passing()
                self.active_constraints.append("min_passing")

        def min_stop(self):
            if "min_stop" in self.active_constraints:
                print("Contraint already active.")
            else:
                self.parent._min_stop()
                self.active_constraints.append("min_stop")

        def single_track(self):
            if "single_track" in self.active_constraints:
                print("Contraint already active.")
            else:
                self.parent._single_track()
                self.active_constraints.append("single_track")

        def min_headway(self):
            if "min_headway" in self.active_constraints:
                print("Contraint already active.")
            else:
                self.parent._min_headway()
                self.active_constraints.append("min_headway")

        def no_early_departure(self):
            if "no_early_departure" in self.active_constraints:
                print("Contraint already active.")
            else:
                self.parent._no_early_departure
                self.active_constraints.append("no_early_departure")
