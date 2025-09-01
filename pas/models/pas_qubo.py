import itertools
import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag
from pas.data.pas_data import PASData
import time
from abstract.models.abstract_model import AbstractModel


class QuboPAS(AbstractModel):

    def __init__(
            self,
            pas_data: PASData,
    ):
        self.jmn_to_q = pas_data.jmn_to_q
        self.q_to_jmn = pas_data.q_to_jmn
        # number of machines
        self.m: int = pas_data.m
        # number of jobs
        self.j: int = pas_data.j
        # Alpha parameter is the scaling between the set-up time term and the
        # job inherent value term
        self.alpha: float = pas_data.alpha
        # Beta parameter is the scaling between the normalization term and the job inherent value term
        self.beta: float = 1.0
        # How much every job takes
        self.processing_times: list[int] = pas_data.processing_times
        # Setup time: for each pair of variables is a j x j matrix - s[i][j] is the setup time from job i to job j
        self.setup_times: list[list[int]] = pas_data.setup_times
        # Job inherent value: j x m  matrix - v[i][m] is the value of job i on machine m
        self.job_values: list[list[int]] = pas_data.job_values
        # Eligible machines: j sets each with the machines that can do job j
        self.eligible_machines: list[set[int]] = [set(machines) for machines in pas_data.eligible_machines]


        self._machine_jobs: list[set[int]] = pas_data.eligible_jobs
        self._jobs_per_machine: npt.NDArray[np.int16] = pas_data.jobs_per_machine
        # Number of variables in the QUBO matrix: number of possible jobs per machine
        # times number of time steps (which is the same)
        self._q: int = pas_data.num_variables

        # Pre-processing of the job values: machines not eligible for a job are
        # assigned a job value of 0 for that specific job
        # machines = set(i for i in range(self.m))
        # for job, job_machines in enumerate(self.eligible_machines):
        #     for machine in machines.difference(job_machines):
        #         self.job_values[job][machine] = 0

        self._eligible_machines_vec = [
            np.array([int(m in eligible_machines) for m in range(self.m)])
            for eligible_machines in self.eligible_machines
        ]


        # Set the penalty values for the constraints:
        self.lambda_3: float = self._calculate_lambda_3()
        self.lambda_4: float = self._calculate_lambda_4()
        self.lambda_5: float = self._calculate_lambda_5()
        # The model in this case is a QUBO matrix
        self.model = self.build_model()



    def _calculate_lambda_3(self) -> float:
        """
        When doing two jobs at the same time, we would have in worst case one set up times less.
        So we need to use the maximal set up time + 1 to find better solutions with this constraint in any case.
        """

        return np.amax(self.setup_times) + 1

    def _calculate_lambda_4(self) -> float:
        """
        When leaving out a step, there would be one set up time less. We have to add 1 to have the constraint stronger
        in any case.
        """

        return np.amax(self.setup_times) + 1

    def _calculate_lambda_5(self) -> float:
        """
        When doing a job two times, the reward from the objective function would be given twice.
        Further, in worst case, we could save two maximal setup times, because the set-up time to the job done twice
        could be zero.
        That means we have to add the maximum job value + two times the maximal set up time + 1
        With objective 3 (normalization) not doing a job gets an additional benefit of in worst case max(p[j])**2
        where p is the array of processing times.
        """

        return self.alpha * np.amax(self.job_values) + 2 * np.amax(self.setup_times) + (self.beta * np.max(self.processing_times)) ** 2 + 400

    def build_model(
            self,
            symmetric: bool = False,
    ) -> npt.NDArray:
        Q = np.zeros((self._q, self._q))

        # Add the first objective related to the job values (maximize job values)
        diag = self._objective_job_values()
        np.fill_diagonal(Q, diag)

        # Add the second objective related to the setup times (minimize setup times)
        Q += self._objective_setup_times()

        # Add the third objective related to the normalization (maximize job values)
        Q += self._objective_normalization()

        # Constraint 5: "All jobs must be processed once"
        Q += self.c5_all_jobs_done_once()

        # Constraint 3 maximum one job can be processed by a machine at any time:
        Q += self.c3_max_one_job_at_a_time()

        # Constraint 4: On any machine a job can be processed at (n+1) th timestep iff a job
        # was processed at the nth timestep
        Q += self.c4_no_timesteps_skipped()

        if symmetric:
            Q = (Q + Q.T) / 2
        return Q

    def get_model(self):
        return self.model
    
    def solve(self, solve_func, **config):

        start_time = time.time()
        answer = solve_func(Q=self.model, **config)
        solve_time = time.time() - start_time

        sample_list = [s for v,s in answer.first.sample.items()]
        solution = self.decode_solution_2(sample_list)
        
        return {"solution": solution, "energy": answer.first.energy, "runtime": solve_time}

    def _objective_job_values(self) -> npt.NDArray:
        """
        Function to calculate the first objective related to the job values.
        It builds accordingly the values along the diagonal and returns the
        whole diagonal array.
        """
        # Todo: see if one can build the diagonal with tensor products (np.kron)
        #       and compare which way is faster
        diag = np.zeros(self._q)
        for job in range(self.j):
            for machine in self.eligible_machines[job]:
                penalty = -self.alpha * self.job_values[job][machine]
                diag[
                    self.jmn_to_q(job, machine, 0): self.jmn_to_q(
                        job, machine, self._jobs_per_machine[machine]
                )
                ] += penalty
        return diag

    def _objective_setup_times(self) -> npt.NDArray:
        objective_setup_times = np.zeros((self._q, self._q))
        # Minimize the setup times
        for job1, job2 in itertools.combinations_with_replacement(range(self.j), 2):
            # We iterate over machines that can perform both jobs (that is why
            # we use the intersection of both sets)
            for machine in self.eligible_machines[job1].intersection(
                    self.eligible_machines[job2]
            ):
                # Only goes until end - 1 because if it is the last job then
                # there is no setup time needed (and n+1 out of scope)
                for n in range(self._jobs_per_machine[machine] - 1):
                    index1 = self.jmn_to_q(job1, machine, n)
                    index2 = self.jmn_to_q(job2, machine, n + 1)
                    objective_setup_times[index1][index2] += self.setup_times[job1][job2]
                    index1 = self.jmn_to_q(job1, machine, n + 1)
                    index2 = self.jmn_to_q(job2, machine, n)
                    objective_setup_times[index1][index2] += self.setup_times[job2][job1]

        return objective_setup_times

    def _objective_normalization(self) -> npt.NDArray:
        # Set the non-diagonal values.
        objective_normalization = np.zeros((self._q, self._q))
        for m in range(self.m):
            indices = [(j, n) for j in self._machine_jobs[m] for n in range(self._jobs_per_machine[m])]
            for (j1, n1), (j2, n2) in itertools.combinations(indices, 2):
                penalty = 2 * self.beta * self.processing_times[j1] * self.processing_times[j2]
                index1 = self.jmn_to_q(j1, m, n1)
                index2 = self.jmn_to_q(j2, m, n2)
                objective_normalization[index1][index2] += penalty
            for j, n in indices:
                penalty = self.beta * self.processing_times[j] ** 2
                index1 = self.jmn_to_q(j, m, n)
                objective_normalization[index1][index1] += penalty
        return objective_normalization

    def c5_all_jobs_done_once(self) -> npt.NDArray:
        block_arrays = []
        for job in range(self.j):
            dim = int(np.dot(self._eligible_machines_vec[job], self._jobs_per_machine))
            block_j = np.full((dim, dim), 2 * self.lambda_5)
            block_j = np.triu(block_j)
            block_arrays.append(block_j)
        constraint_5 = block_diag(*block_arrays)
        np.fill_diagonal(constraint_5, -self.lambda_5)
        return constraint_5

    def c3_max_one_job_at_a_time(self) -> npt.NDArray:
        constraint_3 = np.zeros((self._q, self._q))
        for m in range(self.m):
            for n in range(self._jobs_per_machine[m]):
                for job1, job2 in itertools.combinations(self._machine_jobs[m], 2):
                    constraint_3[self.jmn_to_q(job1, m, n)][self.jmn_to_q(job2, m, n)] += self.lambda_3
        return constraint_3


    def c4_no_timesteps_skipped(self) -> npt.NDArray:
        constraint_4 = np.zeros((self._q, self._q))
        for m in range(self.m):
            # iterator should be only till self._n_machines[0] - 1
            for n in range(self._jobs_per_machine[m] - 1):
                for job1, job2 in itertools.combinations_with_replacement(self._machine_jobs[m], 2):
                    constraint_4[self.jmn_to_q(job1, m, n)][self.jmn_to_q(job2, m, n + 1)] -= self.lambda_4
                    constraint_4[self.jmn_to_q(job1, m, n + 1)][self.jmn_to_q(job2, m, n)] -= self.lambda_4
                for job in self._machine_jobs[m]:
                    constraint_4[self.jmn_to_q(job, m, n + 1)][self.jmn_to_q(job, m, n + 1)] += self.lambda_4
        return constraint_4

    def decode_solution(
            self,
            result: list[bool],
            sample: int = 0,
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        """
        Returns a dictionary where the keys refer to the machines and the results are
        a tuple (j, n) with the job performed and on which timestep n.
        --> format is: {"machine_m": ((job, timestep))}
        Parameters
        ----------
        result: List[bool]
            List which encodes the solution of the problem.
        sample: int, default=0,
            If multiple samples are given this argument specifies which sample to visualize.

        Returns
        -------
        Dict[str, Any]
        The keys are "machine m" where m is the machine number, and the values
        show a tuple (j, n) with the jobs j that should be done on this machine
        on the timesteps n.
        """

        solutions: list[list[tuple[int, int]]] = [[] for _ in range(self.m)]
        for q, variable in enumerate(result):
            if variable == 1:
                j, m, n = self.q_to_jmn(q)
                solutions[m].append((j, n))
        solutions_sorted = [sorted(sol, key=lambda x: x[1]) for sol in solutions]
        solution_dict = {f"machine_{m}": tuple(solutions_sorted[m]) for m in range(self.m)}
        return solution_dict


    def decode_solution_2(
        self,
        result: list[bool],
        sample: int = 0,
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        """
        Returns a dictionary where the keys refer to the machines and the results are
        a tuple (j, n) with the job performed and on which timestep n.
        --> format is: {"machine_m": ((job, timestep))}
        Parameters
        ----------
        result: List[bool]
            List which encodes the solution of the problem.
        sample: int, default=0,
            If multiple samples are given this argument specifies which sample to visualize.

        Returns
        -------
        Dict[str, Any]
        The keys are "machine m" where m is the machine number, and the values
        show a tuple (j, n) with the jobs j that should be done on this machine
        on the timesteps n.
        """

        solutions = {}
        for q, variable in enumerate(result):
            j, m, n = self.q_to_jmn(q)
            solutions[f"x_{j}_{m}_{n}"] = float(variable)
        return solutions

    @property
    def q(self):
        return self._q

    @property
    def machine_jobs(self):
        return self._machine_jobs


