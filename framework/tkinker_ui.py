import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont

from framework.utils import load_use_cases, save_updated_data
from framework.create_init import create_framework_init
from framework.create_venv import create_and_activate_venv


class UI:

    def __init__(self, data):
        self.data = data
        self.use_cases = data["use_cases"]
        self.check_vars = []

    def run(self):

        root = tk.Tk()
        root.title("Use Case Manager")

        # Create a frame to hold the checkboxes
        checkbox_frame = tk.Frame(root)
        checkbox_frame.pack(padx=20, pady=20)

        # Dynamically create checkboxes for each use case
        for use_case in self.use_cases:
            var = tk.BooleanVar()  # Set default value to True (checked)
            checkbox = tk.Checkbutton(checkbox_frame, text=use_case["name"], variable=var)
            checkbox.pack(anchor="w")
            self.check_vars.append(var)

        # Create a button to add a new use case
        add_button = tk.Button(root, text="Add New Use Case",
                               command=lambda: self.open_add_use_case_window(root, checkbox_frame))
        add_button.pack(padx=20, pady=5)

        # Create a button to remove selected use cases
        remove_button = tk.Button(root, text="Remove Selected Use Cases",
                                  command=lambda: self.remove_use_cases(checkbox_frame))
        remove_button.pack(padx=20, pady=5)

        check_all_button = tk.Button(root, text="Select All Use Cases", command=lambda: self.select_all(checkbox_frame))
        check_all_button.pack(padx=20, pady=5)

        # Create a bold font
        bold_font = tkFont.Font(weight="bold")

        # Create a button to install selected use cases
        install_button = tk.Button(root, text="Install Selected Use Cases", command=lambda: self.install_use_cases())
        install_button.pack(padx=20, pady=5)

        # Start the Tkinter event loop
        root.mainloop()

    def install_use_cases(self):
        selected_use_cases = [use_case["name"] for use_case, var in zip(self.use_cases, self.check_vars) if var.get()]

        if selected_use_cases:
            self.data["use_cases"] = [use_case for use_case in self.use_cases if use_case["name"] in selected_use_cases]
            create_framework_init(self.data)
            create_and_activate_venv(self.data)
            messagebox.showinfo("Install", f"Installing use cases: {', '.join(selected_use_cases)}")
        else:
            messagebox.showwarning("Selection Error", "No use cases selected.")

    def remove_use_cases(self, checkbox_frame):
        selected_use_cases = [use_case["name"] for use_case, var in zip(self.use_cases, self.check_vars) if var.get()]

        if selected_use_cases:
            self.data["use_cases"] = [use_case for use_case in self.use_cases if
                                      use_case["name"] not in selected_use_cases]

            save_updated_data()
            self.reload_main_window(checkbox_frame)

            messagebox.showinfo("Remove", f"Removed use cases: {', '.join(selected_use_cases)}")
        else:
            messagebox.showwarning("Selection Error", "No use cases selected for removal.")

    def reload_main_window(self, checkbox_frame):
        for widget in checkbox_frame.winfo_children():
            widget.destroy()

        self.use_cases = load_use_cases()["use_cases"]

        self.check_vars = []
        for use_case in self.use_cases:
            var = tk.BooleanVar(value=True)
            checkbox = tk.Checkbutton(checkbox_frame, text=use_case["name"], variable=var)
            checkbox.pack(anchor="w")
            self.check_vars.append(var)

    def select_all(self, checkbox_frame):
        for widget in checkbox_frame.winfo_children():
            widget.destroy()

        self.check_vars = []
        for use_case in self.use_cases:
            var = tk.BooleanVar(value=True)
            checkbox = tk.Checkbutton(checkbox_frame, text=use_case["name"], variable=var)
            checkbox.pack(anchor="w")
            self.check_vars.append(var)

    def open_add_use_case_window(self, root, checkbox_frame):
        add_window = tk.Toplevel(root)
        add_window.title("Add New Use Case")

        tk.Label(add_window, text="Name").pack()
        name_entry = tk.Entry(add_window)
        name_entry.pack()
        name_entry.insert(0, "Use Case Name")

        tk.Label(add_window, text="Shortcut").pack()
        shortcut_entry = tk.Entry(add_window)
        shortcut_entry.pack()
        shortcut_entry.insert(0, "UC")

        tk.Label(add_window, text="Folder").pack()
        folder_entry = tk.Entry(add_window)
        folder_entry.pack()
        folder_entry.insert(0, "uc")

        tk.Label(add_window, text="Dataclass").pack()
        dataclass_entry = tk.Entry(add_window)
        dataclass_entry.pack()
        dataclass_entry.insert(0, "Data")

        tk.Label(add_window, text="Models (comma-separated)").pack()
        models_entry = tk.Entry(add_window)
        models_entry.pack()
        models_entry.insert(0, "Cplex, Qubo")

        tk.Label(add_window, text="Evaluationclass").pack()
        evaluationclass_entry = tk.Entry(add_window)
        evaluationclass_entry.pack()
        evaluationclass_entry.insert(0, "Evaluation")

        tk.Label(add_window, text="Plottingclass").pack()
        plottingclass_entry = tk.Entry(add_window)
        plottingclass_entry.pack()
        plottingclass_entry.insert(0, "Plot")

        tk.Label(add_window, text="Requirements").pack()
        requirements_entry = tk.Entry(add_window)
        requirements_entry.pack()
        requirements_entry.insert(0, "requirements.txt")

        def add_use_case(add_window):
            new_use_case = {
                "name": name_entry.get(),
                "shortcut": shortcut_entry.get(),
                "folder": folder_entry.get(),
                "dataclass": dataclass_entry.get(),
                "models": models_entry.get().split(","),
                "evaluationclass": evaluationclass_entry.get(),
                "plottingclass": plottingclass_entry.get(),
                "requirements": requirements_entry.get()
            }

            if all(new_use_case.values()):
                self.data['use_cases'].append(new_use_case)
                save_updated_data(self.data)
                messagebox.showinfo("Success", "New use case added successfully!")
                add_window.destroy()
                self.reload_main_window(checkbox_frame)
            else:
                messagebox.showwarning("Input Error", "Please fill out all fields.")

        add_button = tk.Button(add_window, text="Add This Use Case", command=lambda: add_use_case(add_window))
        add_button.pack(pady=10)


