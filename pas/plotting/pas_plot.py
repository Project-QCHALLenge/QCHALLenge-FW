import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from io import BytesIO
from datetime import date, timedelta

from pas.data.pas_data import PASData
from pas.data.utils import SolutionType


TAB10_COLORS = {
    "blue": "#1f77b4",
    "orange" : "#ff7f0e",
    "green" : "#2ca02c",
    "red" : "#d62728",
    "purple" : "#9467bd",
    "brown" : "#8c564b",
    "pink" : "#e377c2",
    "gray" : "#7f7f7f",
    "olive" : "#bcbd22",
    "cyan" : "#17becf"
}


class PASPlot:
    
    def __init__(self, evaluation):
        self.data = evaluation.data
        self.solution = evaluation.solution
    def plot(
            self,
            title: str = "plot",
            show_data: bool = False,
    ):

        data = {}
        index = 0
        init_date = date(2023, 1, 1)
        day = timedelta(days=1)

        for machine, jobs in self.solution.items():
            elapsed_days = timedelta(days=0)
            m = int(machine[-1])
            for job, n in jobs:
                processing_time = self.data.processing_times[job]
                setup_time = 0
                if n > 0:
                    previous_job = None
                    for prev in jobs:
                        if prev[1] == n - 1:
                            previous_job = prev[0]
                    if previous_job is not None:
                        setup_time = self.data.setup_times[previous_job][job]
                start_date = init_date + elapsed_days + day * setup_time
                end_date = start_date + day * (setup_time + processing_time)
                elapsed_days += (end_date - start_date)

                j = "Job " + str(job) + "<br>"
                j += "val:" + str(self.data.job_values[job][m]) + "<br>"
                if show_data:
                    j += "processing time:" + str(self.data.processing_times[job]) + "<br>"
                    if n > 0:
                        j += " set up time:" + str(setup_time)

                data[index] = {
                    "machine": machine,
                    "job": j,
                    "start": start_date,
                    "finish": end_date,
                }
                index += 1
        sol_len = len(data)

        l1 = [
            dict(
                start=data[i]["start"],
                finish=data[i]["finish"],
                machine=data[i]["machine"].capitalize()
                        + "<br>"
                        + str(self.data.eligible_jobs[int(data[i]["machine"][-1:])])[1:-1],
                job=data[i]["job"],
            )
            for i in range(sol_len)
        ]
        df = pd.DataFrame(l1)

        fig = px.timeline(
            df,
            x_start="start",
            x_end="finish",
            y="machine",
            title=title,
            text="job",
            #color_discrete_sequence=[TAB10_COLORS["blue"]],
            color="job"
        )
        fig.update_layout(font={"size": 20}, showlegend=False)
        fig.update_yaxes(
            title_text="Machines",
            title_font={"size": 26},
            autorange="reversed",
            ticklabelposition="inside",
            tickfont={"size": 16},
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_traces(
            textposition="inside",
            textfont_size=14,
            orientation="h",
            insidetextanchor="middle",
        )
        buffer = BytesIO()
        fig.write_image(buffer, format='png')
        buffer.seek(0)
        plt.figure(figsize=(12,8))
        img = mpimg.imread(buffer, format='png')
        plt.imshow(img)
        plt.axis('off')
        return plt

