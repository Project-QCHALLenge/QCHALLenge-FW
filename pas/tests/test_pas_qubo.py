import unittest
from pas.models.pas_qubo import QuboPAS
from pas.data.pas_data import PASData
import numpy as np
from pas.evaluation.evaluation import EvaluationPAS
import gurobipy as gp
from gurobipy import GRB, Var
import dimod



class TestPASQUBO(unittest.TestCase):
    @staticmethod
    def create_gurobi_model_form_Q(Q):
        model = gp.Model("QUBO")
        variables = model.addVars(range(Q.shape[0]), vtype=GRB.BINARY)
        x = gp.MVar.fromlist(list(variables.values()))
        model.setObjective(x.T @ Q @ x, GRB.MINIMIZE)
        return model

    @staticmethod
    def solve_function(Q, *args, **kwargs):
        model = TestPASQUBO.create_gurobi_model_form_Q(Q)
        model.optimize()
        all_vars = model.getVars()
        values = model.getAttr("X", all_vars)
        names = model.getAttr("VarName", all_vars)
        answer = np.zeros(shape=(Q.shape[0]))
        for name, val in zip(names, values):
            index = int(name.replace("C", ""))
            answer[index] = val
        answer_object = dimod.SampleSet.from_samples(answer, "BINARY", model.ObjVal)
        return answer_object

    def test_jobs_on_all_machines_uniform_p_no_setup_only_value(self):
        nr_of_jobs = np.random.randint(2, 5)
        nr_of_machines = np.random.randint(2, 5)
        alpha, beta = 0, 0
        values = np.zeros(shape=(nr_of_jobs, nr_of_machines))
        for m in range(nr_of_machines):
            values[:, m] = m+1
        processing_times = np.zeros(shape=(nr_of_jobs)) + 1
        setup_times = np.zeros(shape=(nr_of_jobs, nr_of_jobs))
        eligible_machines = [range(nr_of_machines) for i in range(nr_of_jobs)]
        instance_dict = {"m": nr_of_machines, "j": nr_of_jobs, "alpha": alpha, "beta": beta, "job_values": values,
                         "processing_times" : processing_times, "setup_times" : setup_times,
                         "eligible_machines": eligible_machines}
        data = PASData(**instance_dict)
        model = QuboPAS(data)
        solution = model.solve(TestPASQUBO.solve_function)["solution"]
        eval_object = EvaluationPAS(data, solution)
        objective = eval_object.get_objective()

        nr_of_violated_constraint = 0
        for constraint, violations in eval_object.check_solution().items():
            if len(violations) > 0:
                nr_of_violated_constraint += 1

        self.assertEqual(nr_of_violated_constraint, 0)


        self.assertEqual(objective, -nr_of_jobs * nr_of_machines)  # add assertion here

    def test_jobs_on_one_machines_uniform_p_uniform_setup_value_and_setup(self):
        nr_of_jobs = np.random.randint(2, 5)
        nr_of_machines = 1
        print(nr_of_jobs, nr_of_machines)
        alpha, beta = 1, 0
        value = np.random.randint(1, 5)
        values = np.zeros(shape=(nr_of_jobs, nr_of_machines)) + value
        processing_times = np.zeros(shape=(nr_of_jobs)) + 1
        setup_times = np.zeros(shape=(nr_of_jobs, nr_of_jobs)) + 1
        eligible_machines = [range(nr_of_machines) for i in range(nr_of_jobs)]
        instance_dict = {"m": nr_of_machines, "j": nr_of_jobs, "alpha": alpha, "beta": beta, "job_values": values,
                         "processing_times" : processing_times, "setup_times" : setup_times,
                         "eligible_machines": eligible_machines}
        data = PASData(**instance_dict)
        model = QuboPAS(data)
        solution = model.solve(TestPASQUBO.solve_function)["solution"]
        eval_object = EvaluationPAS(data, solution)
        objective = eval_object.get_objective()

        nr_of_violated_constraint = 0
        for constraint, violations in eval_object.check_solution().items():
            if len(violations) > 0:
                nr_of_violated_constraint += 1

        self.assertEqual(nr_of_violated_constraint, 0)
        self.assertEqual(objective, (nr_of_jobs - 1) - nr_of_jobs * value)  # add assertion here

    def test_jobs_on_all_machines_uniform_p_uniform_setup_all_objectives(self):
        nr_of_machines = np.random.randint(1, 4)
        nr_of_jobs = nr_of_machines * np.random.randint(1, 4)
        alpha, beta = 1, 1
        values = np.zeros(shape=(nr_of_jobs, nr_of_machines)) + 1
        processing_times = np.zeros(shape=(nr_of_jobs)) + 1
        setup_times = np.zeros(shape=(nr_of_jobs, nr_of_jobs)) + 1
        eligible_machines = [range(nr_of_machines) for _ in range(nr_of_jobs)]
        instance_dict = {"m": nr_of_machines, "j": nr_of_jobs, "alpha": alpha, "beta": beta, "job_values": values,
                         "processing_times" : processing_times, "setup_times" : setup_times,
                         "eligible_machines": eligible_machines}
        data = PASData(**instance_dict)
        model = QuboPAS(data)
        solution = model.solve(TestPASQUBO.solve_function)["solution"]
        eval_object = EvaluationPAS(data, solution)
        objective = eval_object.get_objective()

        nr_of_violated_constraint = 0
        for constraint, violations in eval_object.check_solution().items():
            if len(violations) > 0:
               nr_of_violated_constraint += 1

        self.assertEqual(nr_of_violated_constraint, 0)
        self.assertEqual(objective, (1.0/nr_of_machines)*nr_of_jobs**2-nr_of_machines)  # add assertion here

    def test_jobs_on_all_machines_uniform_p_uniform_setup_all_objectives_fail(self):
        nr_of_machines = 2 # np.random.randint(1, 4)
        nr_of_jobs = 6 #nr_of_machines * np.random.randint(1, 4)
        alpha, beta = 1, 1
        values = np.zeros(shape=(nr_of_jobs, nr_of_machines)) + 1
        processing_times = np.zeros(shape=(nr_of_jobs)) + 1
        setup_times = np.zeros(shape=(nr_of_jobs, nr_of_jobs)) + 1
        eligible_machines = [range(nr_of_machines) for _ in range(nr_of_jobs)]
        instance_dict = {"m": nr_of_machines, "j": nr_of_jobs, "alpha": alpha, "beta": beta, "job_values": values,
                         "processing_times" : processing_times, "setup_times" : setup_times,
                         "eligible_machines": eligible_machines}
        data = PASData(**instance_dict)
        model = QuboPAS(data)
        solution = model.solve(TestPASQUBO.solve_function)["solution"]
        eval_object = EvaluationPAS(data, solution)
        objective = eval_object.get_objective()
        nr_of_violated_constraint = 0
        for constraint, violations in eval_object.check_solution().items():
            if len(violations) > 0:
                nr_of_violated_constraint += 1
        self.assertEqual(nr_of_violated_constraint, 0)

        self.assertEqual(objective, (1.0/nr_of_machines)*nr_of_jobs**2-nr_of_machines)  # add assertion here

    def test_jobs_on_one_machines_uniform_p_uniform_setup_value_and_setup_fail(self):
        nr_of_jobs = 4
        nr_of_machines = 1
        print(nr_of_jobs, nr_of_machines)
        alpha, beta = 1, 0
        value = np.random.randint(1, 5)
        values = np.zeros(shape=(nr_of_jobs, nr_of_machines)) + value
        processing_times = np.zeros(shape=(nr_of_jobs)) + 1
        setup_times = np.zeros(shape=(nr_of_jobs, nr_of_jobs)) + 1
        eligible_machines = [range(nr_of_machines) for i in range(nr_of_jobs)]
        instance_dict = {"m": nr_of_machines, "j": nr_of_jobs, "alpha": alpha, "beta": beta, "job_values": values,
                         "processing_times" : processing_times, "setup_times" : setup_times,
                         "eligible_machines": eligible_machines}
        data = PASData(**instance_dict)
        model = QuboPAS(data)
        solution = model.solve(TestPASQUBO.solve_function)["solution"]
        eval_object = EvaluationPAS(data, solution)
        objective = eval_object.get_objective()

        nr_of_violated_constraint = 0
        for constraint, violations in eval_object.check_solution().items():
            if len(violations) > 0:
                nr_of_violated_constraint += 1

        self.assertEqual(nr_of_violated_constraint, 0)
        self.assertEqual(objective, (nr_of_jobs - 1) - nr_of_jobs * value)  # add assertion here

if __name__ == '__main__':
    unittest.main()
