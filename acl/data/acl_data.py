import json
import random
import os

from abstract.data.abstract_data import AbstractData

import pandas as pd

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path


@dataclass(order=True)
class ACLData(AbstractData):
    seed: int = 1
    K: int = 8

    absl_path = os.path.abspath(__file__)
    transporter_json = os.path.join(os.path.dirname(absl_path), "truck_data.json")

    P: list[int] = field(init=False, repr=False)
    Pl: list[tuple[list]] = field(init=False, repr=False)
    Ph: list[list] = field(init=False, repr=False)
    Pa: list[int] = field(init=False, repr=False)
    Pa_nu: list[int] = field(init=False, repr=False)
    Pa_set: set = field(init=False, repr=False)
    Psp: list[list] = field(init=False, repr=False)
    Pt: list[list] = field(init=False, repr=False)

    L_max: list[float] = field(init=False, repr=False)
    H_max: list[float] = field(init=False, repr=False)
    W_max: float = field(init=False, repr=False)
    wp_max: list[float] = field(init=False, repr=False)
    wa_max: list[float] = field(init=False, repr=False)
    wc_max: list[float] = field(init=False, repr=False)
    wt_max: list[float] = field(init=False, repr=False)
    wl_max: list[float] = field(init=False, repr=False)

    vehicles: list[dict] = field(init=False, repr=False)

    def __post_init__(self):
        random.seed(self.seed)

        self.set_vehicles()
        self.json_trucks()

        self.lr_coefficient = {
            0: [[0.33, 0.34], [0.43, 0.40]],
            1: [[0.30, 0.38], [0.40, 0.35]],
            2: [[0.36, 0.38], [0.43, 0.35]],
        }
        self.hr_coefficient = {
            0: [[0.309, 0.225], [0.317, 0.216]],
            1: [[0.208, 0.182], [0.216, 0.199]],
            2: [[0.259, 0.182], [0.259, 0.199]],
        }

        #both cars face forward when no d variable is provided
        self.lr_coefficient_no_d = {
            0: 0.33,
            1: 0.30,
            2: 0.36,
        }
        self.hr_coefficient_no_d = {
            0: 0.309,
            1: 0.208,
            2: 0.259,
        }

    @classmethod
    def create_problem(cls, num_cars = 10, num_trucks = 1, seed = 1):
        return cls.gen_problem(K=num_cars, num_trucks=num_trucks, seed=seed)

    def to_dict(self):
        def convert(value):
            if isinstance(value, set):
                return list(value)
            elif isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            elif is_dataclass(value):
                return asdict(value)
            else:
                return value

        data = self.__dict__.copy()

        # Ergänze manuell nur die Felder, die NICHT in __dict__ stehen
        data["lr_coefficient"] = convert(self.lr_coefficient)
        data["hr_coefficient"] = convert(self.hr_coefficient)
        data["lr_coefficient_no_d"] = convert(self.lr_coefficient_no_d)
        data["hr_coefficient_no_d"] = convert(self.hr_coefficient_no_d)

        # Filtere vehicle_list raus, behalte vehicles drin
        blacklist = {"vehicle_list"}

        return {
            k: convert(v)
            for k, v in data.items()
            if k not in blacklist
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ACLData":
        instance = cls(seed=data["seed"], K=data["K"])
        instance.P = data["P"]
        instance.Pl = data["Pl"]
        instance.Ph = data["Ph"]
        instance.Pa = data["Pa"]
        instance.Pa_nu = data["Pa_nu"]
        instance.Pa_set = set(data["Pa_set"])
        instance.Psp = data["Psp"]
        instance.Pt = data["Pt"]
        instance.L_max = data["L_max"]
        instance.H_max = data["H_max"]
        instance.W_max = data["W_max"]
        instance.wp_max = data["wp_max"]
        instance.wa_max = data["wa_max"]
        instance.wc_max = data["wc_max"]
        instance.wt_max = data["wt_max"]
        instance.wl_max = data["wl_max"]
        instance.vehicles = data["vehicles"]
        
        def convert_keys(d):
            return {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in d.items()}

        instance.lr_coefficient = convert_keys(data["lr_coefficient"])
        instance.hr_coefficient = convert_keys(data["hr_coefficient"])
        instance.lr_coefficient_no_d = convert_keys(data["lr_coefficient_no_d"])
        instance.hr_coefficient_no_d = convert_keys(data["hr_coefficient_no_d"])
        
        instance.truck_list = data["truck_list"]
        return instance
        

    def get_num_variables(self):
        num_car_on_plattform_variables = len(self.P) * self.K
        num_angle_variables = len(self.Pa)
        num_combine_plattform_variables = len(self.Psp)
        num_direction_variables = len(self.P)
        num_total_variables = num_car_on_plattform_variables + num_angle_variables + num_combine_plattform_variables + num_direction_variables
        return num_total_variables
    
    @classmethod
    def gen_problem(cls, K, seed, num_trucks):
        truck_list = ['truck_1' if i % 2 == 0 else 'trailer_1' for i in range(num_trucks)]
        new_class = cls(K=K, seed=seed)
        new_class.create_predefined_transporters(list_of_trucks=truck_list)
        return new_class

    def set_vehicles(self, info=False):
        vehicle_file = os.path.join(os.path.dirname(self.absl_path), "acl_car_data.csv")
        vehicle_df = pd.read_csv(vehicle_file)
        self.vehicle_list = [vehicle_df.iloc[i].to_dict() for i in range(vehicle_df.shape[0])]
        self.vehicles = [
            self.vehicle_list[i]
            for i in random.sample([j for j in range(vehicle_df.shape[0])], self.K)
        ]
        if info:
            for vehicle in self.vehicles:
                print(f"Height: {vehicle['Height']}, Length: {vehicle['Length']}, Weight: {vehicle['Weight']}")


    @classmethod
    def from_random(cls, K, p, length=4, a_percentage=0.5, sp_percentage=0.5):
 
        new_class = cls(K=K)
        
        new_class.set_vehicles()

        # num_loading_pltfrms_minus_uncompleted is only one less if the remaining pltfrm dont fill a complete loading pltfrm
        num_of_completed_loading_pltfrms = p // length
        num_pltfrms_in_last_ldng_pltfrm = p % length

        new_class.Pl = [[length * i + j for j in range(length)] for i in range(num_of_completed_loading_pltfrms)]
        if num_pltfrms_in_last_ldng_pltfrm!=0:
            new_class.Pl += [[length * (num_of_completed_loading_pltfrms) + j for j in range(num_pltfrms_in_last_ldng_pltfrm)]]

        new_class.Ph = []
        for _p in range(p):
            if (_p//length)%2==0:
                new_class.Ph.append([_p])
            else:
                new_class.Ph[_p%length + length*(_p//(2*length))] += [_p]

        if a_percentage==0:
            new_class.Pa = []
        else:
            pltfrms_with_no_neighbours = {sublist[0] for sublist in new_class.Pl if len(sublist) == 1}
            angleable_pltfrms = [x for x in range(p) if x not in pltfrms_with_no_neighbours]

            new_class.Pa = random.sample(angleable_pltfrms, int(a_percentage*p))
            new_class.Pa.sort()

        new_class.Pa_nu = []
        for i in new_class.Pa:
            for sublist in new_class.Pl:
                if i in sublist:
                    if i == sublist[-1]:
                        new_class.Pa_nu.append(i - 1)
                    else:
                        new_class.Pa_nu.append(i + 1)
                    break

        #self.Pa_nu = [(i-1)%min(p,length) + (i//length)*length if (i+1)%length==0 else (i+1)%min(p,length) + (i//length)*length for i in self.Pa]

        if sp_percentage==0:
            new_class.Psp = []
        else:
            all_consecutive_Psp = [pair for sublist in new_class.Pl for pair in [[sublist[j], sublist[j + 1]] for j in range(len(sublist) - 1)]]

            random.shuffle(all_consecutive_Psp)
            new_class.Psp = []
            used_elements = set()
            for tpl in all_consecutive_Psp:
                if len(new_class.Psp) >= int(sp_percentage*p//2):
                    break
                if tpl[0] not in used_elements and tpl[1] not in used_elements:
                    new_class.Psp.append(tpl)
                    used_elements.update(tpl)

        new_class.Pt = []
        i_idx = 0
        while i_idx < len(new_class.Pl):
            if i_idx + 1 < len(new_class.Pl):
                # Combine current sublist and the next one
                new_class.Pt.append(new_class.Pl[i_idx] + new_class.Pl[i_idx + 1])
                i_idx += 2
            else:
                # If the last sublist is left unpaired, add it as is
                new_class.Pt.append(new_class.Pl[i_idx])
                i_idx += 1
        """if p <= 2*length:
            self.Pt = [[i for i in range(p)]]
        else:
            self.Pt = [[i + 2*length*j for i in range(2*length)] for j in range(num_of_completed_loading_pltfrms // 2)]
            self.Pt += [[i + length * num_of_completed_loading_pltfrms for i in range(p % (num_of_completed_loading_pltfrms * length))]]
        """
        new_class.P = [
            i for i in range(p)
        ]

        new_class.Pa_set = set(new_class.Pa)

        car_weights = [vehicle["Weight"] for vehicle in new_class.vehicles]
        car_lengths = [vehicle["Length"] for vehicle in new_class.vehicles]
        car_heights = [vehicle["Height"] for vehicle in new_class.vehicles]

        new_class.L_max = [round(random.uniform(min(car_lengths), max(car_lengths))*len(pl), 2) for pl in new_class.Pl]
        new_class.H_max = [round(random.uniform(min(car_heights), max(car_heights))*2, 2) for _ in range(len(new_class.Ph))]
        new_class.wp_max = [round(random.uniform(min(car_weights), max(car_weights)+400), 2) for _ in range(p)]
        new_class.wa_max = [round(new_class.wp_max[pltfrm] - random.uniform(min(car_weights), max(car_weights))/10, 2) for pltfrm in new_class.Pa]
        new_class.wc_max = [round(new_class.wp_max[pltfrms[0]]+new_class.wp_max[pltfrms[1]]- random.uniform(min(car_weights), max(car_weights))/20,2) for pltfrms in new_class.Psp]
        new_class.wt_max = [round(sum([new_class.wp_max[pltfrm] for pltfrm in truck_pltfrms]), 2) for truck_pltfrms in new_class.Pt]
        new_class.wl_max = [round(sum([new_class.wp_max[pltfrm] for pltfrm in ldng_pltfrms]), 2) for ldng_pltfrms in new_class.Pl]
        new_class.W_max = round(sum([weights for weights in new_class.wp_max]), 2)

        return new_class

    def create_predefined_transporters(self, list_of_trucks):
        self.truck_list = list_of_trucks

        with open(self.transporter_json, "r") as fp:
            data = json.load(fp)

        list_of_trucks = [data[trck] for trck in list_of_trucks]

        combined_truck = {
            "Pl": [],
            "Ph": [],
            "Pa": [],
            "Pa_nu": [],
            "Psp": [],
            "Pt": [],
            "L_max": [],
            "H_max": [],
            "W_max": 0,
            "wp_max": [],
            "wa_max": [],
            "wc_max": [],
            "wt_max": [],
            "wl_max": [],
        }
        for key in combined_truck.keys():
            if key == "W_max":
                combined_truck[key] += sum(d[key] for d in list_of_trucks)

            elif key in {"Pa", "Pa_nu"}:
                for idx_d, d in enumerate(list_of_trucks):
                    for sublist in d[key]:
                        combined_truck[key] += [
                            x + sum([len(_d["Pt"][0]) for _d in list_of_trucks[:idx_d]])
                            for x in sublist
                        ]

            elif key in {"Pl", "Ph", "Psp", "Pt"}:
                for idx_d, d in enumerate(list_of_trucks):
                    for sublist in d[key]:
                        combined_truck[key].append(
                            [
                                x
                                + sum(
                                    [len(_d["Pt"][0]) for _d in list_of_trucks[:idx_d]]
                                )
                                for x in sublist
                            ]
                        )

            elif key in {
                "L_max",
                "H_max",
                "W_max",
                "wp_max",
                "wa_max",
                "wc_max",
                "wt_max",
                "wl_max",
            }:
                combined_truck[key] += [
                    sublist
                    for idx_d, d in enumerate(list_of_trucks)
                    for sublist in d[key]
                ]
            else:
                exit(f"key = {key} is not par of the truck set")

            self.Pl = combined_truck["Pl"]
            self.Ph = combined_truck["Ph"]
            self.Pa = combined_truck["Pa"]
            self.Pa_nu = combined_truck["Pa_nu"]
            self.Psp = combined_truck["Psp"]
            self.Pt = combined_truck["Pt"]
            self.L_max = combined_truck["L_max"]
            self.H_max = combined_truck["H_max"]
            self.W_max = combined_truck["W_max"]
            self.wp_max = combined_truck["wp_max"]
            self.wa_max = combined_truck["wa_max"]
            self.wc_max = combined_truck["wc_max"]
            self.wt_max = combined_truck["wt_max"]
            self.wl_max = combined_truck["wl_max"]

            self.Pa_set = set(self.Pa)

            self.P = [
                i for i in range(sum([len(sublist) for sublist in combined_truck["Pt"]]))
            ]

    @classmethod
    def scaling_data_creator(cls, k_var, p_linear=True, a_exists=True, sp_exists=True):
        if p_linear:
            p = k_var // 2
        else:
            p = 4

        if a_exists:
            a = p
        else:
            a = 0

        if sp_exists:
            q = p // 2
        else:
            q = 0

        cls.Pl = [[2 * i, 2 * i + 1] for i in range(p // 2)]
        cls.Ph = [[2 * i, 2 * i + 1] for i in range(p // 2)]
        cls.Pa = [i for i in range(a)]
        cls.Pa_nu = [(i + 1) % p for i in range(a)]
        cls.Psp = [[2 * i, 2 * i + 1] for i in range(q)]

        sublist_size = 4
        platform_list = [i for i in range(p)]
        load_pltfm_sublists = -(-len(platform_list) // sublist_size)
        cls.Pt = [
            platform_list[i * sublist_size: (i + 1) * sublist_size]
            for i in range(load_pltfm_sublists)
        ]

        cls.P = [
            i for i in range(p)
        ]

        cls.Pa_set = set(cls.Pa)

        cls.L_max = [10000 for _ in range(len(cls.Pl))]
        cls.H_max = [10000 for _ in range(len(cls.Ph))]
        cls.W_max = 12000
        cls.wp_max = [2500 for _ in range(p)]
        cls.wa_max = [10000 for _ in range(len(cls.Pa))]
        cls.wc_max = [10000 for _ in range(len(cls.Psp))]
        cls.wt_max = [10000 for _ in range(len(cls.Pt))]
        cls.wl_max = [10000 for _ in range(len(cls.Pl))]

        return cls()


    @classmethod
    def from_json(cls, file_path: Path):
        """Load class instance from the data from a json file"""
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return cls(**data)

    def json_trucks(self):
        data = {
            "truck_1": {
                "Pl": [[0, 1], [2, 3]],
                "Ph": [[0, 2], [1, 3]],
                "Pa": [[0, 1, 3]],
                "Pa_nu": [[1, 0, 2]],
                "Psp": [[0, 1], [2, 3]],
                "Pt": [[0, 1, 2, 3]],
                "L_max": [10000, 10000],
                "H_max": [5000, 5000],
                "W_max": 12000,
                "wp_max": [2200, 2200, 2200, 2200],
                "wa_max": [2100, 2100, 2100],
                "wc_max": [2800, 2600],
                "wt_max": [12400],
                "wl_max": [5000, 5000],
            },
            "trailer_1": {
                "Pl": [[0, 1, 2], [3, 4, 5]],
                "Ph": [[0, 3], [1, 4], [2, 5]],
                #"Pa": [[2, 3, 4]],
                "Pa": [],
                #"Pa_nu": [[1, 4, 5]],
                "Pa_nu": [],
                "Psp": [[1, 2], [3, 4], [4, 5]],
                "Pt": [[0, 1, 2, 3, 4, 5]],
                "L_max": [15000, 15000],
                "H_max": [3500, 3600, 3500],
                "W_max": 18600,
                "wp_max": [2200, 2200, 1500, 1500, 1500, 2200],
                #"wa_max": [2100, 2100, 2100],
                "wa_max": [],
                "wc_max": [0, 0, 0],
                "wt_max": [18600],
                "wl_max": [7800, 8000],
            },
            "truck_a": {
                "Pl": [[0, 1], [2, 3]],
                "Ph": [[0, 2], [1, 3]],
                "Pa": [[0, 1, 3]],
                "Pa_nu": [[1, 0, 2]],
                #"Psp": [[0, 1], [2, 3]],
                "Psp": [],
                "Pt": [[0, 1, 2, 3]],
                "L_max": [10000, 10000],
                "H_max": [5000, 5000],
                "W_max": 12000,
                "wp_max": [2200, 2200, 2200, 2200],
                "wa_max": [2100, 2100, 2100],
                #"wc_max": [2800, 2600],
                "wc_max": [],
                "wt_max": [12400],
                "wl_max": [5000, 5000],
            },
            "truck_sp": {
                "Pl": [[0, 1], [2, 3]],
                "Pvehicle_dfh": [[0, 2], [1, 3]],
                #"Pa": [[0, 1, 3]],
                "Pa": [],
                #"Pa_nu": [[1, 0, 2]],
                "Pa_nu": [],
                "Psp": [[0, 1], [2, 3]],
                "Pt": [[0, 1, 2, 3]],
                "L_max": [10000, 10000],
                "H_max": [5000, 5000],
                "W_max": 12000,
                "wp_max": [2200, 2200, 2200, 2200],
                #"wa_max": [2100, 2100, 2100],
                "wa_max": [],
                "wc_max": [2800, 2600],
                "wt_max": [12400],
                "wl_max": [5000, 5000],
            },
            "truck_none": {
                "Pl": [[0, 1], [2, 3]],
                "Ph": [[0, 2], [1, 3]],
                # "Pa": [[0, 1, 3]],
                "Pa": [],
                # "Pa_nu": [[1, 0, 2]],
                "Pa_nu": [],
                #"Psp": [[0, 1], [2, 3]],
                "Psp": [],
                "Pt": [[0, 1, 2, 3]],
                "L_max": [10000, 10000],
                "H_max": [5000, 5000],
                "W_max": 12000,
                "wp_max": [2200, 2200, 2200, 2200],
                # "wa_max": [2100, 2100, 2100],
                "wa_max": [],
                #"wc_max": [2800, 2600],
                "wc_max": [],
                "wt_max": [12400],
                "wl_max": [5000, 5000],
            },
            "truck_2": {
                "Pl": [[0, 1]],
                "Ph": [[0], [1]],
                "Pa": [[0, 1]],
                "Pa_nu": [[1, 0]],
                "Psp": [[0, 1]],
                "Pt": [[0, 1]],
                "L_max": [10000],
                "H_max": [5000, 5000],
                "W_max": 12000,
                "wp_max": [2200, 2200],
                "wa_max": [2100, 2100],
                "wc_max": [2800],
                "wt_max": [12400],
                "wl_max": [5000],
            },
        }

        with open(self.transporter_json, "w") as fp:
            json.dump(data, fp)
