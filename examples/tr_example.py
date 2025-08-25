import bootstrap

from tr.data.tr_data import TRData
from tr.models import TRCplex
from tr.evaluation.evaluation import TREvaluation
from tr.plotting.tr_plot import TRPlot

params = {"stations": 3, "trains": 3}
data = TRData.create_problem(**params)

cplex_model = TRCplex(data)
answer = cplex_model.solve(TimeLimit=1)
print(f"solution: {answer['solution']}")

evaluation = TREvaluation(data, answer["solution"])
print(f"solution clean: {evaluation.solution}")

print(f"objective = {evaluation.get_objective()}")
for constraint, violations in evaluation.check_solution().items():
    if len(violations) > 0:
        print(f"contraint {constraint} was violated {len(violations)} times")

TRPlot(evaluation).plot()
