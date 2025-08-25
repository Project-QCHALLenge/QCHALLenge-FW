import bootstrap

from sp.models import SPCplex
from sp import SPData, SPEvaluation, SPPlot

params = {"lidar_density": 0.1, "street_point_density": 0.1}
data = SPData().create_problem_from_glb_file(**params)
SPPlot(data=data).plot_problem().show()

cplex_model = SPCplex(data)
answer = cplex_model.solve()
print(f"solution: {answer['solution']}")

evaluation = SPEvaluation(data, answer["solution"])
print(f"solution clean: {evaluation.solution}")

print(f"objective = {evaluation.get_objective()}")
for constraint, violations in evaluation.check_solution().items():
    if len(violations) > 0:
        print(f"contraint {constraint} was violated {len(violations)} times")

plt = SPPlot(evaluation).plot(hide_never_covered = True)
plt.show()




