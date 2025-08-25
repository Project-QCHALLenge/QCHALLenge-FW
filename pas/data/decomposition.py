import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
from itertools import combinations, product
from pas.data.pas_data import PASData
import numpy as np
from pas.evaluation.evaluation import SolutionType


def create_problem_graph_machines(pas_data: PASData):
    """
    Creates a graph from the problem data.
    """
    # Initialize an empty graph
    G = nx.Graph()

    # Add nodes for each machine
    for machine in range(pas_data.m):
        G.add_node(machine)

    # For each job, find all pairs of machines that can perform the job
    for job_machines in pas_data.eligible_machines:
        for machine1, machine2 in combinations(job_machines, 2):
            # If an edge already exists between the machines, increment the weight
            if G.has_edge(machine1, machine2):
                G[machine1][machine2]['weight'] += 1
            # If not, add an edge with weight 1
            else:
                G.add_edge(machine1, machine2, weight=1)
    return G


def create_problem_graph_jobs(pas_data: PASData):
    """
    Creates a graph from the problem data where nodes are jobs and edges
    represent the ability of a machine to perform both jobs.
    """
    # Initialize an empty graph
    G = nx.Graph()

    # Add nodes for each job
    for job in range(pas_data.j):
        G.add_node(job)

    # For each machine, find all pairs of jobs that the machine can perform
    for machine in range(pas_data.m):
        eligible_jobs = [job for job in range(pas_data.j) if machine in pas_data.eligible_machines[job]]
        for job1, job2 in combinations(eligible_jobs, 2):
            # If an edge already exists between the jobs, update the weight to
            # the minimum of the current weight and the new weight
            if G.has_edge(job1, job2):
                G[job1][job2]['weight'] = (
                    min(G[job1][job2]['weight'], min(pas_data.setup_times[job1][job2], pas_data.setup_times[job2][job1])
                        ))
            # If not, add an edge with weight equal to min(s[i][j], s[j][i])
            else:
                G.add_edge(job1, job2, weight=min(pas_data.setup_times[job1][job2], pas_data.setup_times[job2][job1]))
    return G




class DataDecomposer:
    """
    This class decomposes a problem instance of PASData into multiple smaller instances to
    be solved separately. The decomposition is based on the problem graph of the instance.
    """

    def __init__(
            self,
            data: PASData,
            max_problem_size: int = 40,
            rank: int = 1,
            map_to_initial_problem: dict[str, dict[int, int]] = None
    ):
        """
        Parameters
        ----------
        data: PASData
            The data of the problem instance to be decomposed.
        max_problem_size: int, default=40
            The maximum number of variables that a sub-problem should contain.
        """
        self.data = data
        self.max_problem_size = max_problem_size
        self.children = []
        self.graph = create_problem_graph_machines(data)
        # Which percentage of the total jobs are affected by the decomposition. This number is useful
        # as a metric to compare different decompositions.
        self.changed_jobs: int
        self.rank = rank
        if map_to_initial_problem is None:
            # Instance is an initial problem, so the mapping is the identity
            self.map_to_initial_problem = {
                "machines": {machine: machine for machine in range(self.data.m)},
                "jobs": {job: job for job in range(self.data.j)}
            }
        else:
            # Instance is a decomposed problem, so the mapping is given by the parent
            # in the self._divide_problem method.
            self.map_to_initial_problem = map_to_initial_problem



    def decompose(self) -> list[tuple["PASData", dict[str, dict[int, int]]]]:
        """
        Decompose the problem instance into smaller sub-problems.
        Returns
        -------
        list[tuple[PASData, dict[str, dict[int, int]]]]
            A list of tuples containing the sub-problems as PASData and the mapping from the sub-problems to
             the parent problem.

        """
        # If the problem is already small enough, return the problem as is.
        if self.data.num_variables <= self.max_problem_size:
            # If the problem is already small enough return (PASData and map_to_parent).
            return [(self.data, self.map_to_initial_problem)]
        elif self.data.m < 4:
            # In general, we don't problems with only one machine because it becomes a simple scheduling problem.
            # so if the problem has 4 or fewer machines, we return the problem as is. As decomposing a pro
            return [(self.data, self.map_to_initial_problem)]
        else:
            # If the problem is too large, decompose it into two smaller sub-problems
            # and call the decompose method recursively on each sub-problem until we are only left
            # with small enough sub-problems.
            subproblem1, subproblem2 = self._divide_problem()
            decomposition = subproblem1.decompose() + subproblem2.decompose()
            return decomposition




    def _divide_problem(
            self
            ) -> tuple["DataDecomposer", "DataDecomposer"]:
        """
        We divide the problem into two sub-problems by finding the minimum cut of the graph,
        so we try to group the machines into two sub-problems such that the sum of the jobs shared between
        the two groups of machines is minimized.
        """

        machines_sub_problem1, machines_sub_problem2 = kernighan_lin_bisection(self.graph)
        jobs_sub_problem1 = set().union(*[self.data.eligible_jobs[machine] for machine in machines_sub_problem1])
        jobs_sub_problem2 = set().union(*[self.data.eligible_jobs[machine] for machine in machines_sub_problem2])
        common_jobs = jobs_sub_problem1.intersection(jobs_sub_problem2)
        self.changed_jobs = len(common_jobs) / self.data.j
        for job in common_jobs:
            job_values = self.data.job_values[job]
            # Extract the index of the machine with the highest job value in job_values
            machine = np.argmax(job_values)
            # The Job goes to the sub-problem with the machine with the highest job value.
            # TODO: Alternatively one could add all the job values of the machines in both sub-problems and
            #    compare that value. The job goes to the sub-problem with the highest sum of job values.
            if machine in machines_sub_problem1:
                jobs_sub_problem2.remove(job)
            else:
                jobs_sub_problem1.remove(job)
        sub_problem1, map_to_parent1 = self.create_sub_problem(machines_sub_problem1, jobs_sub_problem1)
        sub_problem2, map_to_parent2 = self.create_sub_problem(machines_sub_problem2, jobs_sub_problem2)

        # compose the map_to_parent of the sub-problems with the map_to_initial_problem of the parent problem to obtain
        # the map_to_initial_problem of the sub-problems.
        map_to_initial_problem1 = self.compose_mappings(self.map_to_initial_problem, map_to_parent1)
        map_to_initial_problem2 = self.compose_mappings(self.map_to_initial_problem, map_to_parent2)


        # If the problem that is being decomposed already has a map_to_parent object, the map_to_parent
        # of the sub-problems is the composition of the map_to_parent of the parent problem and the map_to_parent of
        # the sub-problem (because in the end the idea is to map to the overall initial problem).
        decomp_1 = DataDecomposer(data=sub_problem1, max_problem_size=self.max_problem_size, rank=self.rank + 1,
                                  map_to_initial_problem=map_to_initial_problem1)
        decomp_2 = DataDecomposer(data=sub_problem2, max_problem_size=self.max_problem_size, rank=self.rank + 1,
                                  map_to_initial_problem=map_to_initial_problem2)
        return decomp_1, decomp_2


    @staticmethod
    def compose_mappings(
            map_parent_to_initial_problem, map_to_parent: dict[str, dict[int, int]]) -> dict[str, dict[int, int]]:
        """
        Compose the map_to_parent of the sub-problems with the map_to_parent of the parent problem.
        We want to map always back to the initial problem, after one decomposition the map_to_parent is the
        same as the map to the initial problem. But if we decompose again, we need to compose the map_to_parent of the
        sub-problems with the map_to_initial_problem of the parent problem to always get back to the very first problem.
        """

        map_to_initial_problem = {
            "machines": {},
            "jobs": {}
        }
        machines = len(map_to_parent["machines"].keys())
        jobs = len(map_to_parent["jobs"].keys())

        for m, j in product(range(machines), range(jobs)):
            map_to_initial_problem["machines"][m] = (
                map_parent_to_initial_problem)["machines"][map_to_parent["machines"][m]]
            map_to_initial_problem["jobs"][j] =\
                map_parent_to_initial_problem["jobs"][map_to_parent["jobs"][j]]

        return map_to_initial_problem




    def create_sub_problem(
            self, machines_sub_problem: set[int], jobs_sub_problem: set[int]
    ) -> tuple[PASData, dict[str, dict[int, int]]]:
        """
        This function takes a set of machines and jobs that define a sub-problem and creates a new PASData object.
        The function also returns a mapping from the indices of the sub-problem to the indices of the parent problem.
        Parameters
        ----------
        machines_sub_problem: machines in the sub-problem
        jobs_sub_problem: jobs in the subproblem

        Returns
        -------
        PASData, dict[str, dict[int, int]]
            The sub-problem data and the mapping from the sub-problem to the parent problem.

        """


        processing_times = [self.data.processing_times[job] for job in jobs_sub_problem]

        # Define the set of jobs and machines that are not in the sub-problem to delete them from
        # the setup times, job values and eligible machines.
        deleted_jobs = {job for job in range(self.data.j)} - jobs_sub_problem
        deleted_machines = {machine for machine in range(self.data.m)} - machines_sub_problem

        # Remove the jobs from the setup times that are not in the sub-problem
        setup_times = np.asarray(self.data.setup_times, dtype=np.int16)
        setup_times = np.delete(setup_times, list(deleted_jobs), axis=0)
        setup_times = np.delete(setup_times, list(deleted_jobs), axis=1)

        # Remove the jobs and machines that are not in the sub-problem from the job values array
        job_values = np.asarray(self.data.job_values, dtype=np.int16)
        job_values = np.delete(job_values, list(deleted_jobs), axis=0)
        job_values = np.delete(job_values, list(deleted_machines), axis=1)

        # Remove the sets for jobs that are not in the sub-problem, and in each set remove the machines that
        # are not in the sub-problem
        eligible_machines = [set(self.data.eligible_machines[job]) - deleted_machines for job in jobs_sub_problem]


        # Create a mapping from the indices of the sub-problem to the indices of the parent problem
        map_to_parent = {
            "machines": {new_index: old_index for new_index, old_index in enumerate(sorted(machines_sub_problem))},
            "jobs": {new_index: old_index for new_index, old_index in enumerate(sorted(jobs_sub_problem))}
        }
        # Inverse map of map_to_parent
        map_to_child = {
            "machines": {v: k for k, v in map_to_parent["machines"].items()},
            "jobs": {v: k for k, v in map_to_parent["jobs"].items()}
        }

        # The eligible_machines object is defined with respect to the parent indices.
        # We need to convert it to the child indices.
        eligible_machines = [
            [map_to_child["machines"][machine] for machine in machines]
            for machines in eligible_machines
        ]

        data_sub_problem = PASData(
            m=len(machines_sub_problem),
            j=len(jobs_sub_problem),
            processing_times=processing_times,
            setup_times=setup_times,
            job_values=job_values,
            eligible_machines=eligible_machines,
            alpha=self.data.alpha
        )
        return data_sub_problem, map_to_parent

    @classmethod
    def map_solution(cls, solution: SolutionType, map_to_parent: dict[str, dict[int, int]]) -> SolutionType:
        """
        Maps a solution from the sub-problem to the parent problem.
        Parameters
        ----------
        solution: SolutionType
            The solution to be mapped.
        map_to_parent: dict[str, dict[int, int]]
            The mapping from the sub-problem to the parent problem.

        Returns
        -------
        SolutionType
            The mapped solution.
        """

        mapped_solution = {}
        for key, value in solution.items():
            try:
                # the key is from the form "machine_m" so the 8th character is the machine number
                old_machine = int(key[8])
                new_machine = map_to_parent["machines"][old_machine]
            except ValueError:
                raise ValueError("Wrong solution format. The key should be of the form 'machine_m'")

            mapped_solution[f"machine_{new_machine}"] = \
                [(map_to_parent["jobs"][job], timestep) for job, timestep in value]
        return mapped_solution





























