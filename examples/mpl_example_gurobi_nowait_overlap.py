import bootstrap

from mpl.data.mpl_data import MPLData
from mpl.models import MPLGurobiNoWaitOverlap
from mpl.evaluation.evaluation import MPLEvaluation
from mpl.plotting.mpl_plot import MPLPlot

params = {"N_A": 1, "N_B": 1, "R": 2, "T": 20, "processing_times": {'Jobs_A':(4,2), 'Jobs_B':(2,4,2)}}
problem = MPLData.create_problem(**params)

gurobi_model = MPLGurobiNoWaitOverlap(problem)
answer = gurobi_model.solve()

evaluation = MPLEvaluation(problem, answer["solution"])
solution = evaluation.solution
print(f"solution clean: {solution}")

print(f"objective = {evaluation.get_objective()}")

fig = MPLPlot(evaluation).plot()
fig.show()