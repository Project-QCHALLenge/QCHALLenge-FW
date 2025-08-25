import bootstrap

from acl.data.acl_data import ACLData
from acl.models import ACLGurobi
from acl.evaluation.evaluation import ACLEvaluation
from acl.plotting.acl_plot import ACLPlot

params = {"num_cars": 8, "seed": 4, "num_trucks": 1}
data = ACLData.create_problem(**params)

cplex_model = ACLGurobi(data)
answer = cplex_model.solve()
print(f"solution: {answer['solution']}")

evaluation = ACLEvaluation(data=data, solution=answer["solution"])
print(f"solution clean: {evaluation.solution}")

print(f"objective = {evaluation.get_objective()}")
for constraint, violations in evaluation.check_solution().items():
    if len(violations) > 0:
        print(f"contraint {constraint} was violated {len(violations)} times")

plt = ACLPlot(evaluation)
plt.show(version=1)