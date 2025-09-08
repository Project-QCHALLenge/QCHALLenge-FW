import unittest
from pas.models.pas_gurobi import GurobiPAS
from pas.data.pas_data import PASData
import numpy as np
from pas.evaluation.evaluation import EvaluationPAS
from pas.plotting.pas_plot import PASPlot


class MyTestCase(unittest.TestCase):
    def test_jobs_on_all_machines_uniform_p_no_setup_only_value(self):
        nr_of_jobs = np.random.randint(2, 10)
        nr_of_machines = np.random.randint(2, 10)
        alpha, beta = 0, 0
        values = np.zeros(shape=(nr_of_jobs, nr_of_machines)) + 1
        processing_times = np.zeros(shape=(nr_of_jobs)) + 1
        setup_times = np.zeros(shape=(nr_of_jobs, nr_of_jobs))
        eligible_machines = [range(nr_of_machines) for i in range(nr_of_jobs)]
        instance_dict = {"m": nr_of_machines, "j": nr_of_jobs, "alpha": alpha, "beta": beta, "job_values": values,
                         "processing_times" : processing_times, "setup_times" : setup_times,
                         "eligible_machines": eligible_machines}
        data = PASData(**instance_dict)
        model = GurobiPAS(data)
        model.solve()
        solution = model.solution
        objective = EvaluationPAS(data, solution).get_objective()

        self.assertEqual(objective, -nr_of_jobs)  # add assertion here

    def test_jobs_on_one_machines_uniform_p_uniform_setup_value_and_setup(self):
        nr_of_jobs = np.random.randint(2, 10)
        nr_of_machines = 1
        alpha, beta = 1, 0
        values = np.zeros(shape=(nr_of_jobs, nr_of_machines)) + 1
        processing_times = np.zeros(shape=(nr_of_jobs)) + 1
        setup_times = np.zeros(shape=(nr_of_jobs, nr_of_jobs)) + 1
        eligible_machines = [range(nr_of_machines) for i in range(nr_of_jobs)]
        instance_dict = {"m": nr_of_machines, "j": nr_of_jobs, "alpha": alpha, "beta": beta, "job_values": values,
                         "processing_times" : processing_times, "setup_times" : setup_times,
                         "eligible_machines": eligible_machines}
        data = PASData(**instance_dict)
        model = GurobiPAS(data)
        model.solve()
        solution = model.solution
        objective = EvaluationPAS(data, solution).get_objective()

        self.assertEqual(objective, -1)  # add assertion here

    def test_jobs_on_all_machines_uniform_p_uniform_setup_all_objectives(self):
        nr_of_machines = 4#np.random.randint(1, 5)
        nr_of_jobs = 8 #nr_of_machines * np.random.randint(1, 5)
        alpha, beta = 1, 1
        values = np.zeros(shape=(nr_of_jobs, nr_of_machines)) + 1
        processing_times = np.zeros(shape=(nr_of_jobs)) + 1
        setup_times = np.zeros(shape=(nr_of_jobs, nr_of_jobs)) + 1
        eligible_machines = [range(nr_of_machines) for _ in range(nr_of_jobs)]
        instance_dict = {"m": nr_of_machines, "j": nr_of_jobs, "alpha": alpha, "beta": beta, "job_values": values,
                         "processing_times" : processing_times, "setup_times" : setup_times,
                         "eligible_machines": eligible_machines}
        data = PASData(**instance_dict)
        model = GurobiPAS(data)
        model.solve()
        solution = model.solution
        objective = EvaluationPAS(data, solution).get_objective()
        plt = PASPlot(EvaluationPAS(data, solution)).plot_solution(title=f"PAS with {data.m} machines and {data.j} jobs")
        plt.show()

        self.assertEqual(objective, (1.0/nr_of_machines)*nr_of_jobs**2-nr_of_machines)  # add assertion here


if __name__ == '__main__':
    unittest.main()
