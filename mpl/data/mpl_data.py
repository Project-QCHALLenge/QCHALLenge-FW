from abstract.data.abstract_data import AbstractData


class MPLData(AbstractData):
    def __init__(self, params):
        # number of A and B type jobs
        self.N_A = params["N_A"]
        self.N_B = params["N_B"]
        # total time
        self.T = params["T"]

        # processing / production time for different job types on different machines
        self.processing_times = params["processing_times"]

        # Machines
        self.M = params["M"]
        self.machine_names = {
            0: "M1_Sack-filling",
            1: "M2_Stretch-hood",
            2: "M3_Oktabin-filling",
            3: "M2_Stretch-hood",
        }
        # time for AGV move
        self.t_r = params["t_r"]

        # number of AGVs
        self.R = params["R"]

        # timlimit for solving
        self.timelimit = params["timelimit"]

        self.__data__ = dict()
        self.__data__["N_A"] = self.N_A
        self.__data__["N_B"] = self.N_B
        self.__data__["T"] = self.T
        self.__data__["processing_times"] = self.processing_times
        self.__data__["M"] = self.M
        self.__data__["machine_names"] = self.machine_names
        self.__data__["t_r"] = self.t_r
        self.__data__["R"] = self.R
        self.__data__["timelimit"] = self.timelimit

    @classmethod
    def create_problem(cls, N_A = 1, N_B = 1, R = 2, T=80, processing_times = {"Jobs_A": (7, 5), "Jobs_B": (4, 6, 4)}, seed = 1):
        params = {"N_A": N_A, "N_B": N_B, "M": 4, "R": R, "t_r": 2, "T": T,
          "processing_times": processing_times,
          "timelimit": 1000}
        print(params)
        return cls(params)

    def to_dict(self):
        return self.__data__.copy()
    
    @classmethod
    def from_dict(cls, data: dict) -> "MPLData":
        params = {
            "N_A": int(data["N_A"]),
            "N_B": int(data["N_B"]),
            "T": int(data["T"]),
            "processing_times": data["processing_times"],
            "M": int(data["M"]) if "M" in data else 4,
            "t_r": int(data["t_r"]) if "t_r" in data else 2,
            "R": int(data["R"]),
            "timelimit": int(data["timelimit"]) if "timelimit" in data else 60
        }
        return cls(params)

    def from_random(cls, *args, **kwargs):
        raise NotImplementedError

    def from_json(cls,  *args, **kwargs):
        raise NotImplementedError

    def get_num_variables(self):
        # based on the formula for theoretical num vars from scaling
        return (self.N_A + self.N_B) * 2 * self.R * 4 * self.T + self.T
