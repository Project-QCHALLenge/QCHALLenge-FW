import matplotlib.pyplot as plt
import os
import pandas as pd


def compare_models(result_file_path: str, colors, offset):
    # Get the current working directory
    current_path = os.getcwd()

    # Get the parent directory (one folder above)
    parent_path = os.path.dirname(current_path)
    file_path = os.path.join(parent_path, result_file_path)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Group by instance_size and solver and calculate the mean and standard deviation for runtime and objective value
    grouped_df = (
        df.groupby(["instance_size", "solver"])
        .agg(
            mean_runtime=("runtime", "mean"),
            std_runtime=("runtime", "std"),
            mean_objective_value=("objective_value", "mean"),
            std_objective_value=("objective_value", "std"),
        )
        .reset_index()
    )

    print(grouped_df.head())

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # Unique instance sizes
    instance_sizes = sorted(grouped_df["instance_size"].unique())

    # Plot for Mean Runtime
    for solver in grouped_df["solver"].unique():
        solver_data = grouped_df[grouped_df["solver"] == solver]
        axs[0].errorbar(
            solver_data["instance_size"] + offset[solver],
            solver_data["mean_runtime"],
            yerr=solver_data["std_runtime"],
            fmt="o",
            capsize=5,
            label=solver,
            color=colors[solver],
        )
        axs[0].plot_solution(
            solver_data["instance_size"] + offset[solver],
            solver_data["mean_runtime"],
            color=colors[solver],
            linestyle="-",
        )

    # Adding labels and title for the first chart
    axs[0].set_xlabel("Instance Size")
    axs[0].set_ylabel("Mean Runtime")
    axs[0].set_title("Mean Runtime vs Instance Size with Error Bars")
    axs[0].legend()
    axs[0].set_xticks(instance_sizes)

    # Plot for Mean Objective Value
    for solver in grouped_df["solver"].unique():
        solver_data = grouped_df[grouped_df["solver"] == solver]
        axs[1].errorbar(
            solver_data["instance_size"] + offset[solver],
            solver_data["mean_objective_value"],
            yerr=solver_data["std_objective_value"],
            fmt="o",
            capsize=5,
            label=solver,
            color=colors[solver],
        )
        axs[1].plot_solution(
            solver_data["instance_size"] + offset[solver],
            solver_data["mean_objective_value"],
            color=colors[solver],
            linestyle="-",
        )

    # Adding labels and title for the second chart
    axs[1].set_xlabel("Instance Size")
    axs[1].set_ylabel("Mean Objective Value")
    axs[1].set_title("Mean Objective Value vs Instance Size with Error Bars")
    axs[1].legend()
    axs[1].set_xticks(instance_sizes)

    # Display the plots
    plt.tight_layout()
    plt.show()
