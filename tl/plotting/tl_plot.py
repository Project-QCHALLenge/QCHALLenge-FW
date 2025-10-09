import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from abstract.plot.abstract_plot import AbstractPlot
from utils.tl_plotting_generic import compare_models

class TL2DPlot(AbstractPlot):

    def __init__(self, evaluation):
        self.data = evaluation.data
        self.selected_boxes = evaluation.solution["solution"]

    def plot_solution(self):

        # Initialize the figure and 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection=None)

        L = self.data.truck_length
        W = self.data.truck_width

        # Draw the container (optional)
        ax.bar(0, 0, L, W, alpha=0.1, color="white", linewidth=1)

        cmap = plt.get_cmap("winter")
        colors = [cmap(i) for i in np.linspace(0, 1, len(self.data.boxes))]
        
        # Iterate over each box to plot it
        for i, box_placing in self.selected_boxes.items():
            # Get dimensions of the box
            idx, length, width, weight, area = self.data.boxes.loc[i, ['index', 'length', 'width', 'weight', 'area']]

            # Get coordinates of the lower-left-front corner of the box
            x_coord = box_placing["x_coord"]
            y_coord = box_placing["y_coord"]

            # get the value of r
            r_values = box_placing["rotation"]

            # calculate new width length height after rotation
            n_length = length * (1 - r_values) + width * r_values
            n_width = length * r_values + width * (1 - r_values)

            # Draw the box
            rect = patches.Rectangle(
                (x_coord, y_coord),
                n_length,
                n_width,
                color=colors[i],
                linewidth=1,
                alpha=0.9,
            )
            ax.add_patch(rect)

            ax.text(
            x_coord + n_length / 2, 
            y_coord + n_width / 2, 
            str(idx), 
            color="white", 
            fontsize=10, 
            ha="center", 
            va="center",
            fontweight="bold",
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3')
            )

        # Set labels and title
        ax.set_xlabel("X Axis - Truck Length")
        ax.set_ylabel("Y Axis - Truck Width")

        # ax.set_zlabel('Z Axis')
        ax.set_title("Boxes in Container")

        # Set the limits of the axis to the container size
        ax.set_xlim([0, L])
        ax.set_ylim([0, W])
        ax.set_aspect("equal")
        # ax.set_zlim([0, H])

        # Set the aspect ratio to be equal
        return plt


def visualize_time_evaluation(result_file_path: str):

    # Define colors for each solver
    colors = {
        "Gurobi": "#FF3333",
        "Cplex": "#0530AD",
        "Scip": "grey",
        "DantzigWolfe": "green",
    }

    # Define an offset for each solver
    offset = {"Gurobi": -0.3, "Cplex": -0.1, "Scip": 0.1, "DantzigWolfe": 0.3}

    compare_models(result_file_path=result_file_path, colors=colors, offset=offset)
