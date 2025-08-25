import bootstrap

from sp.models import SPQuboBinary, SPQAOA
from sp import SPData, SPEvaluation, SPPlot

params = {"version": 1, "num_cols": 2, "max_radius": 2.5}
data = SPData().create_problem(**params) 

qubo_model = SPQuboBinary(data, 10, 10, 22)
qaoa_model = SPQAOA(QuboModel=qubo_model, layers=2, type="binary")

answer = qaoa_model.solve(iterations=100, optimizer="Adam", learning_rate=0.01, seed=501, info=True)
print(f"solution = {answer['solution']}")

evaluation = SPEvaluation(data, answer['solution'])
print(f"solution clean: {evaluation.solution}")

print(f"objective = {evaluation.get_objective()}")
for constraint, violations in evaluation.check_solution().items():
    if len(violations) > 0:
        print(f"contraint {constraint} was violated {len(violations)} times")

plt = SPPlot(evaluation).plot(hide_never_covered = True)
plt.show()