import pandas
import numpy as np
import plotly.express as px
from lb.data.lb_data import LBData
from lb.evaluation.evaluation import LBEvaluation


class LBPlot:
    def __init__(self, data : LBData, evaluation : LBEvaluation):
        self.data = data
        self.evaluation = evaluation
        self.solution_as_data_frame = self.evaluation.interpret_solution()

    def plot_solution(self):
        # Plotly Timeline plots based on a DF, where each row represents one unit assigned to some truck.
        # Instead of time, weight will be plotted for each unit per truck.
        # Therefore, the interpreted solution gets converted in this format.
        list_of_dicts = []

        # Build the mentioned structure truck by truck, i.e. for each row of the solution.
        for row in self.solution_as_data_frame.iloc:
            # Converts solution on the current truck to a list of tuples (product_weight, product_type).
            # This tuple is added for each unit of that type on this truck.
            list_of_product_and_load = [(int(row[f"LoadOfProduct_{i}"] / row[f"NumberOfUnitsProduct_{i}"]),
                                         i)
                                        for i in self.data.product_types
                                        for _ in range(int(row[f"NumberOfUnitsProduct_{i}"]))
                                        if row[f"NumberOfUnitsProduct_{i}"] != 0]

            # Sort the list of tuples by weight, since in the plot we want the units to be sorted by weight.
            list_of_product_and_load = np.array(sorted(list_of_product_and_load, key=lambda x: x[0], reverse=True))
            # Calculate end of bar representing a unit by the cumsum of the weights.
            end_times = np.cumsum(list_of_product_and_load[:, 0])
            # Start of bars are end of bars, but shifted. Start of the first bar on each truck is 0.
            start_times = np.insert(end_times[:-1], 0, 0)

            # Creates  dictionaries with said structure.
            list_of_dicts += [dict(Truck=row["Truck"], Start=start_times[i], Finish=end_times[i],
                                   ProductType=list_of_product_and_load[:, 1][i])
                              for i in range(list_of_product_and_load.shape[0])]

        # Converts list of dictionaries to DF
        df = pandas.DataFrame(list_of_dicts)

        # Timeline natively works only for timestamps.
        # Here we have integers representing weight of unit.
        # Calculate weight per unit so this information can be injected into the plot later.
        df['delta'] = df['Finish'] - df['Start']

        # Standard usage of the timeline plot.
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Truck", text="ProductType", color="ProductType")

        # Creates vertical line in plot, representing average weight per truck.
        fig.add_vline(x=sum(self.solution_as_data_frame["TotalLoad"]) / len(self.solution_as_data_frame.index), line_width=3, line_color="black")
        fig.update_yaxes(autorange="reversed")

        fig.layout.xaxis.type = 'linear'

        # Workaround for the timestamp issue
        fig.data[0].x = df.delta.tolist()

        fig.update_layout(xaxis_title="Weight")

        # Label y-axis with "Truck1" ... "TruckN"
        fig.update_yaxes(
            tickvals=self.solution_as_data_frame["Truck"],  # positions of the ticks
            ticktext=[f"Truck{i}" for i in self.solution_as_data_frame["Truck"]]  # labels for each position
        )

        return fig








