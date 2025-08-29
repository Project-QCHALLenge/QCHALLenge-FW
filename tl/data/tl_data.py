import pandas as pd
from dataclasses import dataclass
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)
# local import
from tl.utils.tl_data_generic import TruckLoadingData, TruckParameters
from abstract.data.abstract_data import AbstractData


@dataclass(order=True)
class TLData(TruckLoadingData, AbstractData):
    """
    Dataclass for the 2 dimensional truckloading problem (tl2d)
    """

    def __init__(self, truck_parameters: TruckParameters, boxes: pd.DataFrame):
        super().__init__(boxes, truck_parameters)
        self.num_variables =(len(boxes) # number of box origins
                            * truck_parameters.truck_length * truck_parameters.truck_width # possible box origins in the truck
                            * 2 # rotation
                            ) 
        
    @classmethod
    def create_problem(cls,
        stations: int = 2,
        trains: int = 2,
        connectivity: float = 0.5,
        seed: int = None,
        min_path_len: int = 1,
        method: str = "gnp",):
        return cls.from_random(stations, trains, connectivity, seed, min_path_len, method)
    
    @classmethod
    def create_problem(cls, num_boxes: int = 5, seed: int = 1):
        return cls.get_random_problem(num_boxes, seed)

    def to_dict(self):
        return {
            "truck_parameters": self.truck_parameters.to_dict() if hasattr(self.truck_parameters, "to_dict") else str(self.truck_parameters),
            "truck_length": int(self.truck_length),
            "truck_width": int(self.truck_width),
            "truck_height": int(self.truck_height),
            "truck_capacity": int(self.truck_capacity),
            "boxes": self.boxes.to_dict(orient="records") if hasattr(self.boxes, "to_dict") else str(self.boxes),
            "num_variables": int(self.num_variables)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TLData":
        truck_parameters = TruckParameters(
            truck_length=data["truck_length"],
            truck_width=data["truck_width"],
            truck_height=data["truck_height"],
            truck_capacity=data["truck_capacity"]
        )

        boxes = pd.DataFrame(data["boxes"])

        instance = cls(
            truck_parameters=truck_parameters,
            boxes=boxes
        )

        instance.truck_length = data["truck_length"]
        instance.truck_width = data["truck_width"]
        instance.truck_height = data["truck_height"]
        instance.truck_capacity = data["truck_capacity"]
        instance.num_variables = data["num_variables"]

        return instance

    @classmethod
    def from_random(cls,  num_boxes, seed: int = 1):
        return cls.get_random_problem(num_boxes, seed)

    @classmethod
    def get_random_problem(cls, num_boxes, seed: int = 1):
        """
        Generates a random problem dataset with the specified number of boxes and a given random seed.

        This method creates a set of truck parameters based on the number of boxes provided, and generates
        a list of boxes using the specified random seed for reproducibility. The method then returns an
        instance of the class containing these parameters and the generated boxes.

        :param num_boxes: The number of boxes to generate.
        :param random_seed: An integer seed for random number generation to ensure reproducibility.
        :return: An instance of the class containing the generated truck parameters and boxes.
        """

        
        boxes = cls.generate_box(
            num_boxes=num_boxes,
            random_seed=seed,
        )
        area = sum(boxes.area)
        # truck length is box volume / 2m with a width of 2m (real value is ca. 2.5) and heigth of 3m (real value is ca. 3m)
        truck_parameters = TruckParameters(
            truck_length=int(area / 2),
            truck_width=2,
            truck_height=3,
            truck_capacity=int(area) * 3,
        )
        # generate data object
        tl2d_data = cls(
            truck_parameters=truck_parameters,
            boxes=boxes,
        )

        return tl2d_data
        
    def get_num_variables(self):
        return self.num_variables

    @classmethod
    def from_json(cls,  *args, **kwargs):
        raise NotImplementedError
