#Careful! Consumes a lot of DWave Time

import bootstrap

from mpl.data.mpl_data import MPLData
from mpl.models import MPLCQMNoWaitOverlap
from mpl.evaluation.evaluation import MPLEvaluation
from mpl.plotting.mpl_plot import MPLPlot

params = {"N_A": 1, "N_B": 1, "R": 2, "T": 30, "processing_times": {'Jobs_A':(7,5), 'Jobs_B':(4,6,4)}}
problem = MPLData.create_problem(**params)

config = {"timelimit": 5}
gurobi_model = MPLCQMNoWaitOverlap(problem)
answer = gurobi_model.solve(**config)

evaluation = MPLEvaluation(problem, answer["solution"])
solution = evaluation.solution
print(f"solution clean: {solution}")

print(f"objective = {evaluation.get_objective()}")

fig = MPLPlot(evaluation).plot_solution()
fig.show()