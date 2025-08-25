import bootstrap

from lb.models.lb_dwave_cqm_unary_uniform_truck_capacity import LBDWAVECQMUnaryUniformTruckCapacity
from dwave.samplers import SimulatedAnnealingSampler
from lb.data.lb_data import LBData
from lb.evaluation.evaluation import LBEvaluation
from lb.plotting.lb_plot import LBPlot


# Define problem instance
number_of_trucks = 4
# [(quantity_1, weight_1), ..., (quantity_n, weight_n)]
products = [(5, 20), (4, 76), (9, 23), (2, 24)]
data = LBData.create_problem(number_of_trucks, products)

# Create gurobi model
model = LBDWAVECQMUnaryUniformTruckCapacity(data)
# Solve problem
solve_func = SimulatedAnnealingSampler().sample
model.solve(solve_func, num_reads=1000)

# Fetch solution
solution = model.solution()
print(solution)
# Evaluate solution
eval_object = LBEvaluation(solution, data)
# Check feasibility
print(eval_object.check_solution())
# Solution as Dataframe
solution_as_data_frame = eval_object.interpret_solution()
print(solution_as_data_frame)
# Objective Value
print(eval_object.get_objective())


# Plot solution
plot = LBPlot(data, eval_object).plot_solution()
plot.show()
