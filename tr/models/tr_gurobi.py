import gurobipy as gp
from gurobipy import GRB, Var
from abstract.models.abstract_model import AbstractModel
from functools import cached_property
import time

from tr.data.tr_data import TRData
from tr.models.tr_cplex import TR_cplex
from dimod import (
    Vartype,
)

from transformations.from_cplex import FromCPLEX


class GurobiTR(TR_cplex, AbstractModel):

    def __init__(self, data: TRData):
        super().__init__(data)

        self._grb_model = FromCPLEX(self._model).to_gurobi()
        self._bqm, self._inverter = FromCPLEX(self._model).to_bqm()

    def solve(self, **config):
        TimeLimit = config.get("TimeLimit", 30)
        self._grb_model.Params.TimeLimit = TimeLimit

        start_time = time.perf_counter()
        self._grb_model.optimize()
        runtime = time.perf_counter() - start_time

        solution = {}
        for var in self._grb_model.getVars():
            solution.update({var.varName: int(var.x)})

        return {
            "solution": solution,
            "energy": 0,
            "runtime": runtime,
            "num_variables": self._grb_model.NumVars,
        }

    def solve_bqm(
        self,
        quiet: bool = False,
        gap: float = None,
        work_limit: float = None,
        objective_stop: float = None,
        seed: int = None,
        start: dict[str, int] = None,
        **config,
    ) -> dict:

        timeout = config.get("TimeLimit", 30)
        qm = self._bqm
        qm.num_variables
        inverter = self._inverter
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", int(not quiet))
        env.start()
        gm = gp.Model(env=env)

        # Translate the binary quadratic model as objective with just binary
        # variables
        if qm.vartype == Vartype.SPIN:
            qm = qm.change_vartype(Vartype.BINARY, inplace=False)
        vardict = {}
        for var in qm.variables:
            vardict[var] = gm.addVar(name=str(var), vtype=GRB.BINARY)

        if start is not None:
            for var, v in start.items():
                vardict[var].Start = v

        obj = sum(vardict[name] * bias for name, bias in qm.iter_linear()) + qm.offset
        obj += sum(
            vardict[n1] * vardict[n2] * bias for n1, n2, bias in qm.iter_quadratic()
        )

        gm.setObjective(obj)

        # Set Gurobi solver parameters
        if timeout is not None:
            gm.setParam("TimeLimit", timeout)
        if work_limit is not None:
            gm.setParam("WorkLimit", work_limit)
        if objective_stop is not None:
            gm.setParam("BestObjStop", objective_stop)
        if gap is not None:
            gm.setParam("MIPGap", gap)
        if seed is not None:
            gm.setParam("Seed", seed)

        # Solve
        start_time = time.perf_counter()
        gm.optimize()
        runtime = time.perf_counter() - start_time

        bqm_solution = {k: int(round(v.X)) for k, v in vardict.items()}

        solution = inverter(bqm_solution)

        return {
            "solution": solution,
            "energy": 0,
            "runtime": runtime,
            "num_variables": qm.num_variables,
        }

    # @cached_property
    # def solution(self):
    #     solution = {}
    #     for var, val in self.variables.items():
    #         if isinstance(val, Var):
    #             solution[var] = int(val.X)
    #         else:
    #             solution[var] = val
    #     return solution
