import bootstrap

from sp.models import SPQuboBinary
from sp import SPData, SPEvaluation, SPPlot

import neal

params = {"version": 3, "num_cols": 5, "max_radius": 2.5}
data = SPData().create_problem(**params) 

#solve problem
config = {"num_reads":1000,"num_sweeps":1000}
solve_func = neal.SimulatedAnnealingSampler().sample_qubo

qubo_model_bin = SPQuboBinary(data)
answer = qubo_model_bin.solve(solve_func, **config)
print(f"solution = {answer['solution']}")

evaluation = SPEvaluation(data, answer['solution'])
print(f"solution clean: {evaluation.solution}")

print(f"objective = {evaluation.get_objective()}")
for constraint, violations in evaluation.check_solution().items():
    if len(violations) > 0:
        print(f"contraint {constraint} was violated {len(violations)} times")

plt = SPPlot(evaluation).plot(hide_never_covered = True)
plt.show()




