import random
import json

import numpy as np
import numpy.typing as npt

from typing import Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy

@dataclass(order=True)
class PASData:
    """ Dataclass for the PAS data."""
    num_variables: int = field(init=False, repr=False)
    m: int
    j: int
    alpha: int
    processing_times: list[int]
    setup_times: list[list[int]]
    job_values: list[list[int]]
    eligible_machines: list[list[int]]
    beta: int = 1
    eligible_jobs: list[set[int]] = field(init=False, repr=False)
    jobs_per_machine: npt.NDArray[np.int16] = field(init=False, repr=False)
    eligible_machines_vec: list[npt.NDArray[int]] = field(init=False, repr=False)

    def __post_init__(self):
        self.eligible_jobs = [set(job for job, machine_sets in enumerate(self.eligible_machines) if machine in machine_sets) for machine in range(self.m)]
        self.jobs_per_machine = np.array([len(self.eligible_jobs[machine]) for machine in range(self.m)], dtype=np.int32)
        # number of variables is summing #jobs and #timesteps per machine over all machines.
        # #of timesteps per machine is the same as the  of #jobs per machine so:
        self.num_variables = np.dot(self.jobs_per_machine, self.jobs_per_machine)
        self.eligible_machines_vec = [
            np.array([int(m in eligible_machines) for m in range(self.m)])
            for eligible_machines in self.eligible_machines
        ]

    @classmethod
    def create_problem(cls, m: int = 2, j: int = 4, alpha: float = 1.0, seed: int = None):
        return cls.from_random(m,j,alpha,seed)

    def to_dict(self):
        def convert(value):
            if isinstance(value, set):
                return list(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, list):
                return [convert(v) for v in value]
            elif isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            else:
                return value

        return {
            "m": self.m,
            "j": self.j,
            "alpha": self.alpha,
            "processing_times": convert(self.processing_times),
            "setup_times": convert(self.setup_times),
            "job_values": convert(self.job_values),
            "eligible_machines": convert(self.eligible_machines),
            "beta": self.beta,
            "eligible_jobs": convert(self.eligible_jobs),
            "jobs_per_machine": convert(self.jobs_per_machine),
            "eligible_machines_vec": convert(self.eligible_machines_vec),
            "num_variables": convert(self.num_variables),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PASData":
        instance = cls(
            m=data["m"],
            j=data["j"],
            alpha=data["alpha"],
            processing_times=data["processing_times"],
            setup_times=data["setup_times"],
            job_values=data["job_values"],
            eligible_machines=data["eligible_machines"],
            beta=data.get("beta", 1)
        )

        instance.eligible_jobs = [set(jobs) for jobs in data["eligible_jobs"]]
        instance.jobs_per_machine = np.array(data["jobs_per_machine"], dtype=np.int32)
        instance.eligible_machines_vec = [np.array(arr) for arr in data["eligible_machines_vec"]]
        instance.num_variables = data["num_variables"]

        return instance

    def get_num_variables(self):
        return self.num_variables

    @classmethod
    def from_json(cls, file_path: Path):
        """ Load the data from a json file"""
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return cls(**data)

    @classmethod
    def from_random(cls, m: int, j: int = None, alpha: float = 1.0, seed: int = None):
        """ Create a random instance of the problem using Abhisheks funtion. """
        # Set the seed for the random number generator
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # use the create_random_data function to generate random data
        data = cls.create_random_data(m, j, alpha)
        return cls(**data)

    def jmn_to_q(self, j: int, m: int, n: int) -> int:
        """
        Help function, takes as input the job number, the machine number and the time slot and gives the q index for
        the corresponding variable.
        Parameters
        ----------
        j: int
            Job number.
        m: int
            Machine number. Has to be a valid machine e.g. one of the eligible machines for job j.
        n: int
            Time slot for machine m.

        Returns
        -------
        int
        Index corresponding to the specified variable.
        """
        q = 0
        # job part: number of variables after j - 1 jobs
        for job in range(j):
            q += np.dot(self.eligible_machines_vec[job], self.jobs_per_machine)
        # machine part: number of variables for job j after m - 1 machines
        if m not in self.eligible_machines[j]:
            raise ValueError(
                f"The selected machine {m} is not eligible for job {j}. "
                f"Job {j} can only be performed in machines {self.eligible_machines[j]}"
            )
        q += np.dot(self.eligible_machines_vec[j][:m], self.jobs_per_machine[:m])
        if n > self.jobs_per_machine[m]:
            raise ValueError(
                f"The selected timestep {n} is out of scope for machine {m}. "
                f"The selected machine can only do {self.jobs_per_machine[m]} jobs."
            )
        q += n
        return q


    def q_to_jmn(self, q: int) -> tuple[int, int, int]:
        """
        Help function. Takes the variable index q and returns the corresponding job number, machine number and time
        step.
        Parameters
        ----------
        q: int
        Index for the desired variable.

        Returns
        -------
        Tuple[int, int, int]
        Job number, machine number and time step.
        """
        # We first find the value for j.
        j = 0
        # total_machine_count: This vector keeps track in the first while loop of how many of the jobs until j have
        # machine m as eligible. For example if there are 3 machines and 7 jobs and the iteration is at the point
        # where j = 5, then if the vector looks like this: total_machine_count = (1, 4, 2) it means that taking
        # into account only the first five jobs, machine 1 is eligible for 1 of the job, machine 2 is eligible
        # for 4 jobs and machine 3 is eligible for 2 jobs.
        total_machine_count = deepcopy(self.eligible_machines_vec[0])
        k = 0
        while k == 0:
            # This
            if q < np.dot(total_machine_count, self.jobs_per_machine):
                break
            else:
                j += 1
                total_machine_count += self.eligible_machines_vec[j]
        q -= np.dot((total_machine_count - self.eligible_machines_vec[j]), self.jobs_per_machine)

        # Set the value for m
        machines = sorted(self.eligible_machines[j])
        m = machines.pop(0)
        var = self.jobs_per_machine[m]
        while k == 0:
            if q < var:
                break
            else:
                m = machines.pop(0)
                var += self.jobs_per_machine[m]

        n = q - (var - self.jobs_per_machine[m])
        return j, m, n

    @classmethod
    def create_random_data(cls, m: int, j: int = None, alpha: float = 1.0
                        ) -> Dict[Any, Any]:
        """
        Abhisheks funtion to generate random problem instances.
        Parameters
        ----------
        m: int
            number of machines.
        j: int
            number of jobs.
        alpha: float, default=1.0
            parameter to scale the second objective.
        """

        if j is None:
            j = int(m+7.5)
        processing_times = np.random.randint(2,10,j)
        setup_times = np.random.randint(2, 10, (j, j))
        job_values = np.random.randint(2, 10, (j, m))
        eligible_machines = []
        for i in range(j):
            num_eligibles = random.randrange(1, m+1)
            eligible = random.sample(range(0, m), num_eligibles)
            eligible_machines.append(eligible)

        # create data dict to store all the values:
        data = {"m": m, "j": j, "alpha": alpha, "processing_times": processing_times.tolist(),
                "setup_times": setup_times.tolist(), "job_values": job_values.tolist(),
                "eligible_machines": eligible_machines}
        return data

