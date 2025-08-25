from abc import ABC

class TruckLoadingModel(ABC):
    model_name: str

    def toggel_output_logging(self, **kwargs):
        print("Output toggeling is not implemented yet!")

    def set_time_limit(self, **kwargs):
        print(f"Model {self.model_name} has no option to insert time limit")

    def optimize(self):
        print("Model has no optimize function, please add.")

    def build_model(self):
        return NotImplementedError("No function to build model is provided!")

    def solve(self, solve_func = None, **config):
        if solve_func:
            self.optimize(solve_func, **config)
        else:
            self.optimize(**config)
        sol = self.generate_solution_overview()

        return sol.interpretation_dict
