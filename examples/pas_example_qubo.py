import bootstrap

from neal import SimulatedAnnealingSampler

from pas.data.pas_data import PASData
from pas.models import PASQubo
from pas.plotting.pas_plot import PASPlot
from pas.evaluation.evaluation import EvaluationPAS

# define the problem size
params = {"m": 4, "j": 6, "seed": 3}
# create a random problem
problem = PASData.create_problem(**params)

# create a qubo
qubo_model = PASQubo(problem)
solve_func = SimulatedAnnealingSampler().sample_qubo
config = {"num_reads": 10000, "num_sweeps": 10000}

# solve the qubo with the solve function
answer = qubo_model.solve(solve_func, **config)
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
plt = PASPlot(eval).plot_solution(title=f"PAS with {problem.m} machines and {problem.j} jobs")
plt.show()











