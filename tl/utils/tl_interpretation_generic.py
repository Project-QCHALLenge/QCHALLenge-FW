from abc import ABC, abstractmethod

from typing import Dict

from .tl_data_generic import TruckLoadingData

class LB_Evaluator_TruckLoading:
    """
    Evaluation of a solution to the Truck Loading problem.
    """ 
    

    def __init__(self, data: TruckLoadingData, solution):
        self.data = data
        self.solution = solution
    
    def get_objective(self):
        """returns the free unused space in the truck."""
        
        return self.solution["objective_value"]
    
    def check_constraints(self):
        raise NotImplementedError("Not yet implemented.")


class Tl_Generic_Solution(ABC):
    """
    This class contains all the __init__ and run information for a specific optimization.
    """

    # {Box 1: {x_coordinate: int, y_coord: int, rotation: bool}, Box 1: {..}}
    model_name: str
    selected_boxes: Dict[str, Dict[str, int]]
    runtime: float
    num_vars: int
    num_constrs: int
    objective_value: float
    violated_constraints: list  # if the list is empty, the solution is feasible
    gap_to_optimal: float  # if the gap is 0, then the solution is optimal
    data: TruckLoadingData

    def __init__(
        self,
        model_name,
        selected_boxes,
        runtime,
        num_vars,
        num_constrs,
        obj_val,
        data,
        violated_constraints,
        gap_to_optimal,
        energy = 0,
    ):
        self.model_name = model_name
        self.selected_boxes = selected_boxes
        self.runtime = runtime
        self.num_vars = num_vars
        self.num_constrs = num_constrs
        self.objective_value = obj_val
        self.data = data
        self.gap_to_optimal = gap_to_optimal
        self.violated_constraints = violated_constraints
        self.energy = energy
        self.interpretation_dict = dict(
            solution=dict(
                model_name=self.model_name,
                num_vars=self.num_vars,
                solution=self.selected_boxes,
                num_constrs=self.num_constrs,
                objective_value=-self.objective_value,
                gap_to_optimal=self.gap_to_optimal,
                violated_constraints=self.violated_constraints,
            ),
            runtime=self.runtime,
            energy=self.energy,
        )

    
    @abstractmethod
    def to_dict(self):
        raise NotImplementedError("to Dict method not implemented.") 


    def __str__(self):
        return f"Solution value found in {self.runtime:.2f} seconds with {self.objective_value:.0f}"

    def __repr__(self):
        return f"Solution value found in {self.runtime:.2f} seconds with {self.objective_value:.0f}"
