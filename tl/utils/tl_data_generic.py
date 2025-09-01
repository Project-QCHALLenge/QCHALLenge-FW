import pandas as pd
import numpy as np
from abstract.data.abstract_data import AbstractData


class TruckParameters(AbstractData):

    truck_length = int
    truck_width = int
    truck_height = int
    truck_capacity = int

    def __init__(self, truck_length, truck_width, truck_height, truck_capacity):
        self.truck_length = truck_length
        self.truck_width = truck_width
        self.truck_height = truck_height
        self.truck_capacity = truck_capacity


class TruckLoadingData(AbstractData):
    boxes: pd.DataFrame
    truck_parameters: TruckParameters
    truck_length: int
    truck_width: int
    truck_height: int
    truck_capacity: int

    def __init__(self, boxes: pd.DataFrame, truck_parameters: TruckParameters):
        self.truck_parameters = truck_parameters
        self.truck_length = truck_parameters.truck_length
        self.truck_width = truck_parameters.truck_width
        self.truck_height = truck_parameters.truck_height
        self.truck_capacity = truck_parameters.truck_capacity
        self.boxes = boxes

    @staticmethod
    def generate_box(
        num_boxes,
        random_seed: int = 1234,
    ):
        """
        This function generates a box around the provided number of boxes and provides the data as a pandas df.
        :param num_boxes:       The number of boxes that should be generated.
        :param random_seed:     The seed for the random dimension generator of the boxes.
        :return:                Pandas dataframe.
        """
        df = []
        np.random.seed(random_seed)

        for i in range(num_boxes):
            length = np.random.randint(1, 3)
            width = np.random.randint(1, 3)
            height = np.random.randint(1, 3)
            weight = np.random.randint(1, 3)

            # we want to have the boxes such that lengt >= width >= height
            if height >= width:
                height, width = width, height
            if width >= length:
                length, width = width, length

            box = [
                i,
                length,
                width,
                height,
                weight,
                length * width,
                length * width * height,
            ]
            df.append(box)

        dataset = pd.DataFrame(df)
        dataset = dataset.rename(
            columns={
                0: "index",
                1: "length",
                2: "width",
                3: "height",
                4: "weight",
                5: "area",
                6: "volume",
            },
            errors="raise",
        )
        return dataset

    def get_box_names_as_list(self):
        return self.boxes["index"].to_list()

    @classmethod
    def load_from_csv(cls, csv_file):
        raise NotImplementedError("CSV loading function is not yet implemented.")

    @classmethod
    def get_example(cls, i: int):
        """
        Return a specified example instance by its index.

        This method is intended to retrieve a specific example of a problem instance based on the index provided.
        However, it is currently not implemented and will raise a NotImplementedError if called.

        :param i: The index of the example instance to retrieve.
        :type i: int
        :raises NotImplementedError: Always raised as this function is not yet implemented.
        :return: None
        """
        raise NotImplementedError(
            "Get example loading function is not yet implemented."
        )

    def get_random_problem(self, **kwargs):
        """
        Returns a specified instance number based on the provided keyword arguments.

        This method is intended to retrieve a specific problem instance, but it is currently
        not implemented. The function raises a `NotImplementedError` to indicate that the
        actual loading mechanism has not been defined yet.

        :param kwargs: Additional keyword arguments that specify the parameters required
                       to identify and retrieve the problem instance.
        :raises NotImplementedError: Indicates that the method is not yet implemented.
        """
        raise NotImplementedError(
            "Get example loading function is not yet implemented."
        )
    
    def get_num_variables(self):
        raise NotImplementedError(
            "Get number of variables (complexity) function is not yet implemented."
        )
