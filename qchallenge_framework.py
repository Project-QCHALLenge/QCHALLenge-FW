import json
from pathlib import Path
from framework.init import model_classes

class QCHALLengeFramework():

    def __init__(self):
        self.use_cases = list(model_classes.keys())
        json_path = Path("framework") / "framework_config.json"
        with json_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        self.use_cases_names = [uc["name"] for uc in data["use_cases"]]
        self.model_classes = model_classes
        print(self.model_classes)

    def get_use_cases(self):
        return self.use_cases
    
    def get_use_case_names(self):
        return self.use_cases_names
    
    def get_data_class(self, use_case):
        return self.model_classes[use_case]["data"]

    def get_model_class(self, use_case, model_name, problem):
        return self.model_classes[use_case][model_name](problem)

    def get_evaluation_class(self, use_case, problem, solution):
        return self.model_classes[use_case]["evaluation"](problem, solution)

    def get_plotting_class(self, use_case, evaluation):
        return self.model_classes[use_case]["plot"](evaluation)
    
