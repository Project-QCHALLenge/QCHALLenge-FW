import json

from pathlib import Path
from tkinter import messagebox


def load_use_cases():
    try:
        json_path = Path("framework") / "framework_config.json"
        with open(json_path, 'r') as file:
            data = json.load(file)
            return data
    except Exception as e:
        # print("Error", f"Failed to load use cases: {e}")
        return {
            "use_cases": []
        }


def save_updated_data(data):
    try:
        json_path = Path("framework") / "framework_config.json"
        with open(json_path, 'r') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save changes: {e}")