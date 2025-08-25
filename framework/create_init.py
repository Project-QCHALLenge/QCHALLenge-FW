from pathlib import Path


def create_framework_init(data):
    framework_init_imports = ""
    model_classes = ""
    for use_case in data["use_cases"]:
        shortcut = use_case['shortcut']
        dataclass = use_case['dataclass']
        models = use_case['models']
        evaluationclass = use_case['evaluationclass']
        plotclass = use_case['plotclass']
        tab = "    "
        model_class = f"""\n{tab}"{shortcut}": {{\n{tab}{tab}"data": {dataclass},\n{tab}{tab}"evaluation": {evaluationclass},\n{tab}{tab}"plot": {plotclass}"""

        for model in models:
            model_key = model.replace(shortcut, "").lower() + "_model"
            model_class += f',\n{tab}{tab}"{model_key}":{model}'

        model_class += f"\n{tab}}},"

        framework_init_imports += f"""from {use_case['folder']} import *\n"""

        model_classes += model_class

    framework_init_content = f"""{framework_init_imports}\nmodel_classes = {{{model_classes}\n}}"""

    path = Path("framework") / "init.py"
    # Write the content to framework_init.py
    with open(path, 'w') as file:
        file.write(framework_init_content.strip())

    print("framework_init.py created successfully.")