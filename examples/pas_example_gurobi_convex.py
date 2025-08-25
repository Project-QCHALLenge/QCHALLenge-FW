import bootstrap

from pas.data.pas_data import PASData
from pas.models import PASGurobiConvex
from pas.plotting.pas_plot import PASPlot
from pas.evaluation.evaluation import EvaluationPAS

# define the problem size
params = {"m": 4, "j": 8, "seed": 1}
# create a random problem
problem = PASData.create_problem(**params)

# create a convex gurobi Model and solve it
gc_model = PASGurobiConvex(problem)
answer = gc_model.solve()
print(f"solution: {answer['solution']}")

# The Evaluation class converts the solution given by the different solvers in a standardized format.
eval = EvaluationPAS(problem, answer["solution"])
solution = eval.solution
print(f"solution clean: {solution}")

print(f"objective = {eval.get_objective()}")
for constraint, violations in eval.check_solution().items():
    if len(violations) > 0:
        print(f"contraint {constraint} was violated {len(violations)} times")

# create a plot
plt = PASPlot(eval).plot(title=f"PAS with {problem.m} machines and {problem.j} jobs")
plt.show()
