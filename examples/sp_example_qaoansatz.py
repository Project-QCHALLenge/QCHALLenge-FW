import bootstrap

from sp import SPData, SPQaoansatz, SPEvaluation, SPPlot

params = {"version": 1, "num_cols": 2, "max_radius": 2.5}
data = SPData().create_problem(**params) 

qaoa_model = SPQaoansatz(data, layers=2)

answer = qaoa_model.solve(iterations=10, optimizer="Adam", learning_rate=0.1, info=True)
print(f"solution = {answer['solution']}")

evaluation = SPEvaluation(data, answer['solution'])
print(f"solution clean: {evaluation.solution}")

print(f"objective = {evaluation.get_objective()}")
for constraint, violations in evaluation.check_solution().items():
    if len(violations) > 0:
        print(f"contraint {constraint} was violated {len(violations)} times")

plt = SPPlot(evaluation).plot_solution(hide_never_covered = True)
plt.show()