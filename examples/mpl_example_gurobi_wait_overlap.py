import bootstrap

from mpl.data.mpl_data import MPLData
from mpl.models import MPLGurobiWaitOverlap
from mpl.evaluation.evaluation import MPLEvaluation
from mpl.plotting.mpl_plot import MPLPlot

params = {"N_A": 1, "N_B": 0, "R": 1, "T":12, "processing_times": {'Jobs_A':(3,1), 'Jobs_B':(1,3,1)}}
problem = MPLData.create_problem(**params)

gurobi_model = MPLGurobiWaitOverlap(problem)
answer = gurobi_model.solve()

evaluation = MPLEvaluation(problem, answer["solution"])
solution = evaluation.solution
print(f"solution clean: {solution}")

print(f"objective = {evaluation.get_objective()}")
for constraint, violations in evaluation.check_solution().items():
    if len(violations) > 0:
        print(f"contraint {constraint} was violated {len(violations)} times")

fig = MPLPlot(evaluation).plot()
fig.show()