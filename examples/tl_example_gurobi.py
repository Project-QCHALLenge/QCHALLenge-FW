import bootstrap

from tl import TLData
from tl import TLGurobi
from tl import TLEvaluation
from tl import TLPlot

params = {"num_boxes": 6, "seed": 3}
data = TLData.create_problem(**params)
print(data.boxes)
print(data.truck_parameters.__dict__)

gurobi_model = TLGurobi(data)
answer = gurobi_model.solve(**{"TimeLimit": 10})
print(f"solution: {answer['solution']}")

evaluation = TLEvaluation(data=data, solution=answer)
for constraint, violations in evaluation.check_solution().items():
    if len(violations) > 0:
        print(f"contraint {constraint} was violated {len(violations)} times")

plt = TLPlot(evaluation).plot_solution()
plt.show()
