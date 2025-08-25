import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from io import BytesIO

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class MPLPlot:

    def __init__(self, evaluation):
        self.df = evaluation.solution
        self.data = evaluation.data
        self.params = self.data.__data__
        self.JOBS_A = list(range(self.params["N_A"]))
        self.JOBS_B = list(range(self.params["N_A"], self.params["N_A"] + self.params["N_B"]))
        self.t_r = self.params["t_r"]
        self.total_jobs = self.JOBS_A + self.JOBS_B
        self.MACHINES = list(range(self.data.M))
        self.total_machines = len(self.MACHINES)
        self.total_jobs = len(self.JOBS_A) + len(self.JOBS_B)
        self.machine_names = self.data.machine_names

    def plot(self):
        # Ensure the 'Start' and 'Finish' columns are numeric
        self.df['Start'] = pd.to_numeric(self.df['Start'], downcast='integer')
        self.df['Finish'] = pd.to_numeric(self.df['Finish'], downcast='integer')

        # Sort machines: AGVs first, then other machines
        self.df['is_agv'] = self.df['Machines'].str.contains('AGV')
        self.df = self.df.sort_values(by=['is_agv', 'Machines', 'Start'], ascending=[False, True, True])

        # Dynamically assign colors for each unique JobName
        unique_jobs = self.df['JobName'].unique()
        colors = list(mcolors.TABLEAU_COLORS.values())  # Use a set of distinguishable colors
        job_colors = {job: colors[i % len(colors)] for i, job in enumerate(unique_jobs)}

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each machine's tasks
        for machine in self.df['Machines'].unique():
            machine_data = self.df[self.df['Machines'] == machine]
            for i, row in machine_data.iterrows():
                # Choose color based on JobName
                color = job_colors[row['JobName']]
                # Plot barh for the task
                ax.barh(machine, row['Finish'] - row['Start'], left=row['Start'], color=color, edgecolor='black')
                
                # Truncate long process labels if necessary
                truncated_process = (row['Process'][:10] + '...') if len(row['Process']) > 10 else row['Process']

                # Add text label for the Process
                ax.text((row['Start'] + row['Finish']) / 2, machine, truncated_process, va='center', ha='center', color='black', fontsize=8)

        # Create a dynamic legend for the jobs
        legend_handles = [mpatches.Patch(color=color, label=job) for job, color in job_colors.items()]
        ax.legend(handles=legend_handles, title="Job Names")

        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Machines')
        ax.set_title(f'Job Schedule by Machine (AGVs First) - {len(unique_jobs)} Jobs, {len(self.df["Machines"].unique())} Machines')

        # Add vertical lines for each time tick (every unit of time)
        max_time = self.df['Finish'].max()
        for t in range(0, max_time + 1):
            ax.axvline(x=t, color='grey', linestyle='--', linewidth=0.5)

        return plt
    
    def plot(self):
        # Ensure the 'Start' and 'Finish' columns are numeric
        self.df['Start'] = pd.to_numeric(self.df['Start'], downcast='integer')
        self.df['Finish'] = pd.to_numeric(self.df['Finish'], downcast='integer')

        # Sort machines: AGVs first, then other machines
        self.df['is_agv'] = self.df['Machines'].str.contains('AGV')
        self.df = self.df.sort_values(by=['is_agv', 'Machines', 'Start'], ascending=[False, True, True])

        # Dynamically assign colors for each unique JobName
        unique_jobs = self.df['JobName'].unique()
        colors = px.colors.qualitative.Plotly  # Use Plotly's qualitative color set from px
        job_colors = {job: colors[i % len(colors)] for i, job in enumerate(unique_jobs)}

        # Create the plot
        fig = go.Figure()
        
        # Plot each machine's tasks
        jobs_in_legend = set()
        for machine in self.df['Machines'].unique():
            machine_data = self.df[self.df['Machines'] == machine]
            for i, row in machine_data.iterrows():
                # Choose color based on JobName
                color = job_colors[row['JobName']]
                # Plot bar for the task
                fig.add_trace(go.Bar(
                    x=[row['Finish'] - row['Start']], 
                    y=[machine], 
                    base=[row['Start']],
                    orientation='h',
                    marker=dict(color=color, line=dict(color='black', width=1)),
                    name=row['JobName'],
                    hoverinfo='text',
                    text=f"{row['Process'][:10]}..." if len(row['Process']) > 10 else row['Process'],
                    hovertext=f"Start: {row['Start']}, Finish: {row['Finish']}, Process: {row['Process']}",
                    showlegend=row['JobName'] not in jobs_in_legend  # Only show legend once per job
                ))
                jobs_in_legend.add(row['JobName'])

        # Update layout for labels, title, and ticks
        fig.update_layout(
            barmode='stack',
            title=f'Job Schedule by Machine (AGVs First) - {len(unique_jobs)} Jobs, {len(self.df["Machines"].unique())} Machines',
            xaxis_title='Time',
            yaxis_title='Machines',
            yaxis=dict(type='category', categoryorder='array', categoryarray=self.df['Machines'].unique()),  # Ensure proper ordering of machines
            showlegend=True,
            legend_title="Job Names",
            hovermode='closest',
            width=1000,
            height=600
        )

        # Add vertical lines for each time tick (every unit of time)
        max_time = self.df['Finish'].max()
        for t in range(0, max_time + 1):
            fig.add_vline(x=t, line=dict(color='grey', dash='dash'), layer='below', line_width=0.5)

        buffer = BytesIO()
        fig.write_image(buffer, format='png')
        buffer.seek(0)
        plt.figure(figsize=(12,8))
        img = mpimg.imread(buffer, format='png')
        plt.imshow(img)
        plt.axis('off')

        return plt

    # def plot(self):

    #     print(self.df)

    #     self.df['Start'] = pd.to_numeric(self.df['Start'], downcast='integer')
    #     self.df['Finish'] = pd.to_numeric(self.df['Finish'], downcast='integer')

    #     fig = px.timeline(self.df, x_start="Start", x_end="Finish", color="JobName", y="Machines", text='Process', height=3 * 100)
    #     fig.layout.xaxis.type = 'linear'

    #     print(fig.data[0].x)  # Check if the x values are correct and integers

    #     count = 0
    #     for n in range(len(self.JOBS_A) + len(self.JOBS_B)):
    #         length = len(fig.data[n].y)
    #         fig.data[n].x = self.df.delta.tolist()[count:count + length]
    #         count += length

    #     print(fig.data[0].x)

    #     fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
    #     fig.update_traces(textposition='inside')
    #     fig.update_layout(title=f'{self.total_machines - 1} Machines, {self.total_jobs} Jobs', xaxis_title="Time (seconds)")
    #     fig.update_yaxes(autorange="reversed", categoryorder='category ascending')

    #     buffer = BytesIO()
    #     fig.write_image(buffer, format='png')
    #     buffer.seek(0)
    #     plt.figure(figsize=(12,8))
    #     img = mpimg.imread(buffer, format='png')
    #     plt.imshow(img)
    #     plt.axis('off')

    #     return plt

    # def plot_plotly(self):
        
    #     df = self.df

    #     fig = px.timeline(df, x_start="Start", x_end="Finish", color="JobName", y="Machines", text='Process', height=3*100)
    #     fig.layout.xaxis.type = 'linear' # type: ignore
    #     count=0
    #     for n in range(len(self.JOBS_A)+len(self.JOBS_B)):
    #         length=len(fig.data[n].y)
    #         fig.data[n].x = df.delta.tolist()[count:count+length]
    #         count+=length

    #     fig.update_layout(xaxis = dict( tickmode = 'linear', tick0 = 0, dtick = 1 ))
    #     fig.update_traces(textposition='inside')
    #     fig.update_layout(title=f'{self.total_machines-1} Machines, {self.total_jobs} Jobs', xaxis_title="Time (seconds)")
    #     fig.update_yaxes(autorange="reversed", categoryorder='category ascending')
        
    #     buffer = BytesIO()
    #     fig.write_image(buffer, format='png')
    #     buffer.seek(0)
    #     plt.figure(figsize=(12,8))
    #     img = mpimg.imread(buffer, format='png')
    #     plt.imshow(img)
    #     plt.axis('off')

    #     return plt
