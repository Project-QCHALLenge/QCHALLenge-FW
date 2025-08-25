from docplex.mp.model import Model
import numpy as np
from dimod import (
    Binary,
    SampleSet,
    quicksum,
)
from copy import deepcopy
import time
from transformations.from_cplex import FromCPLEX

# from neal import SimulatedAnnealingSampler
# from dwave.system import DWaveSampler, EmbeddingComposite
import dimod


class TR_cplex:

    def __init__(self, data, tau_stop: int = 0):
        self.data = data
        self.tau_int = tau_stop
        self.constraints = self.Constraints(self)
        self._solution = None
        self.railnetwork = data.railnet
        self.trains = self.railnetwork._trains  # dict {train.id : Train}
        if len(self.trains) == 0:
            raise ValueError("There are no trains in the network.")
        self._d_max = self.railnetwork.dmax
        self._model = Model(name="TrainScheduling")
        self.violations = dict()
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
            (j, j_1): self._common_subsequents(self.C[j], self.C[j_1])
            for j in self.trains.keys()
            for j_1 in self.trains.keys()
            if j != j_1 and len(self._common_subsequents(self.C[j], self.C[j_1])) > 0
        }

        # pairs of subsequent stations j and j_1 have in common
        # {(1,2):(('C','D'), ('D','F'))}

        self.J_o = {
            (j, j_1): self._common_opposites(self.C[j], self.C[j_1])
            for j in self.trains.keys()
            for j_1 in self.trains.keys()
            if j != j_1 and len(self._common_opposites(self.C[j], self.C[j_1])) > 0
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
        self.num_variables = len(self.t) + len(self.y) + len(self.z)

    def bqm(self):
        self.T = {
            (j, s): [self.upsilon[j, s, "out"] + i for i in range(self._d_max)]
            for j in self.trains.keys()
            for s in self.S[j]
        }
        self._x = {
            (j, t, s): Binary(f"x_{j}_{t}_{s}")
            for j in self.trains.keys()
            for s in self.S[j]
            for t in self.T[j, s]
        }

        # one-hot
        sum_penalty = 1
        objective = sum_penalty * quicksum(
            quicksum() - quicksum() for j in self.trains.keys() for s in self.S[j]
        )

        bqm = dimod.BinaryQuadraticModel("BINARY")
        j = 1
        s = "A"

        for t in self.T[j, s]:
            bqm.add_interaction(f"x_{j}_{t}_{s}", f"x_{1}_{t}_{'C'}", 1.0)
        print(bqm, objective)

    def get_model(self):
        return self._model

    def set_dmax(self, dmax):
        self.data.railnet.dmax = self._d_max + 1
        self.__init__(
            self.data, self.tau_int
        )  # or rebuild self.t which is probably the correct way to do it

    # def set_bqm(self):
    #     self.bqm, self.inverter = FromCPLEX(self._model).to_bqm()

    def solve(self, solver="cplex", quiet=False, **params):
        if solver == "cplex":
            start_timer = time.perf_counter()
            self._solution = self._model.solve(**params)
            runtime = time.perf_counter() - start_timer
            if self._solution is None:
                print(f"Unable to solve: increasing dmax to {self._d_max+1}")
                self.set_dmax(dmax=self._d_max)
                self.solve(solver, quiet, **params)

            vars = {
                var.name: self._solution.get_value(var)
                for var in self._model.iter_variables()
            }
            return {
                "solution": vars,
                "energy": 0,
                "runtime": runtime,
                "num_variables": self.num_variables,
            }

    def solve_qubo(self, solve_func, **config):

        qubo, converter = FromCPLEX(self._model).to_matrix()

        start_time = time.time()
        answer = solve_func(Q=qubo, **config)
        runtime = time.time() - start_time

        solution = converter(answer.first.sample)

        return {
            "solution": solution,
            "energy": answer.first.energy,
            "runtime": runtime,
            "num_variables": len(qubo),
        }

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

    def _common_subsequents(self, tup1, tup2):
        pairs = list()
        for s in tup1:
            if s in tup2:
                pairs.append(s)
        return tuple(pairs)

    def _common_opposites(self, tup1, tup2):
        pairs = list()
        for a, b in tup1:
            if (b, a) in tup2:
                pairs.append((a, b))
        return tuple(pairs)

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
        self.constraints.all()

    @classmethod
    def _make_dicts(cls, solution):
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

    #### constraints and violation checking

    def check_violation(self, solution: dict[str, float] = None, constraint=None):
        """Checks whether a given variables assignment violates contraints."""
        violations = {"precedence y violation": [], "precedence z violation": []}

        t, y, z = self._make_dicts(solution)

        for (j, j1, station), v in y.items():
            if y[j, j1, station] != 1 - y[(j1, j, station)]:
                if (
                    "y",
                    j,
                    j1,
                    station,
                    y[(j, j1, station)],
                    y[j1, j, station],
                ) not in violations["precedence y violation"]:
                    violations["precedence y violation"].append(
                        ("y", j1, j, station, y[(j1, j, station)], y[j, j1, station])
                    )

        for (j, j1, s1, s2), v in z.items():
            if z[j, j1, s1, s2] != 1 - z[j1, j, s2, s1]:
                if (
                    "z",
                    j,
                    j1,
                    s1,
                    s2,
                    z[j, j1, s1, s2],
                    z[j1, j, s2, s1],
                ) not in violations["precedence z violation"]:
                    violations["precedence z violation"].append(
                        ("z", j1, j, s2, s1, z[j1, j, s2, s1], z[j, j1, s1, s2])
                    )
        if constraint is None:
            for constraint in self.constraints.active_constraints:

                violations[constraint] = self._check_constraint(constraint, t, y, z)

            return violations
        else:
            self._check_constraint(constraint, t, y, z)

    def _check_constraint(self, constraint, t, y, z):

        if constraint == "min_passing":
            return self._check_min_passing(t, y, z)
        elif constraint == "min_headway":
            return self._check_min_headway(t, y, z)
        elif constraint == "single_track":
            return self._check_single_track(t, y, z)
        elif constraint == "min_stop":
            return self._check_min_stop(t, y, z)
        elif constraint == "no_early_departure":
            return self._check_no_early_departure(t, y, z)
        else:
            raise ValueError("Constraint not found")

    def _min_passing(self):
        for j, train in self.trains.items():
            for s, s_next in self.C[j]:
                self._model.add_constraint(
                    self.t[j, s_next, "in"] - self.t[j, s, "out"]
                    == self.tau_pass[j, s, s_next],
                    ctname=f"min_passing_ilp_{j}_{s}_{s_next}",
                )

    def _check_min_passing(self, t, y, z):
        violations = []
        for j, train in self.trains.items():

            for s, s_next in self.C[j]:
                if t[j, s_next, "in"] - t[j, s, "out"] != self.tau_pass[j, s, s_next]:
                    self.violations[(j, s, s_next)] = "min_passing"
                    violations.append((j, s, s_next))
        return violations

    def _min_headway(self):
        # min headway constraint
        M = 1000  # the big M to implement implications
        violations = []
        for (j, j_1), tups in self.J_d.items():
            for s, s_next in tups:
                self._model.add_constraint(
                    self.t[j_1, s, "out"]
                    + M * (1 - self.y[j, j_1, s])
                    - self.t[j, s, "out"]
                    >= self.tau_pass[j, s, s_next],
                    ctname=f"min_headway_ilp_{j}_{j_1}_{s}_{s_next}",
                )
        return violations

    def _check_min_headway(self, t, y, z):
        M = 1000
        violations = []
        for (j, j_1), tups in self.J_d.items():
            for s, s_next in tups:
                condition = (
                    t[j_1, s, "out"] + M * (1 - y[j, j_1, s]) - t[j, s, "out"]
                    >= self.tau_pass[j, s, s_next]
                )
                if not condition:
                    if (j_1, j, s) not in self.violations:
                        self.violations[(j, j_1, s)] = "min_headway"
                        violations.append((j, j_1, s))
        return violations

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
        violations = []
        for (j, j_1), tups in self.J_o.items():
            for s, s_next in tups:
                condition = (
                    t[j_1, s_next, "out"] + M * (1 - z[j, j_1, s, s_next])
                    >= t[j, s_next, "in"]
                )
            if not condition:
                if (j_1, j, s_next, s) not in self.violations:
                    self.violations[j, j_1, s, s_next] = "single_track"
                    violations.append((j, j_1, s, s_next))
        return violations

    def _min_stop(self):
        for j, train in self.trains.items():
            for s in train.route:
                self._model.add_constraint(
                    self.t[j, s, "out"] - self.t[j, s, "in"] >= 1,  # hard coded TODO
                    ctname=f"min_stop_{j}_{s}",
                )

    def _check_min_stop(self, t, y, z):
        violations = []
        for j, train in self.trains.items():
            for s in train.route:
                if t[j, s, "out"] - t[j, s, "in"] < 1:  # hard coded TODO
                    self.violations[j, s] = "min_stop"
                    violations.append((j, s))
        return violations

    # no early departure constraint
    def _no_early_departure(self):
        for j, train in self.trains.items():
            for s in self.S[j]:
                self._model.add_constraint(
                    self.t[j, s, "out"] >= self.sigma[j, s, "out"],
                    ctname=f"no_early_departure_{j}_{s}",
                )

    def _check_no_early_departure(self, t, y, z):
        violations = []
        for j, train in self.trains.items():
            for s in self.S[j]:
                if t[j, s, "out"] < self.sigma[j, s, "out"]:
                    self.violations[j, s] = "no_early_departure"
                    violations.append((j, s))
        return violations

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
