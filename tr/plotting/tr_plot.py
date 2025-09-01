from matplotlib import pyplot as plt
from abstract.plot.abstract_plot import AbstractPlot
from tr.evaluation.evaluation import TREvaluation
from tr.data.railnetwork import RailNetwork
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import networkx as nx
import matplotlib.patheffects as patheffects


class TRPlot(AbstractPlot):
    def __init__(self, evaluation : TREvaluation) -> None:
        solution = evaluation.solution
        self.vars: dict = {
            tuple(k[2:].split("_")): v for k, v in solution.items() if k[0] == "t"
        }
        self.railnet: RailNetwork = evaluation.data.railnet
        self.clean: list = self._clean_solution(solution)
        self.t_max = int(max(self.vars.values()))
        self.time_dict_for_step: dict = self.build_timedict(self.clean)
        self.current_plot = None
        self.current_t = 0 

    def _clean_solution(self, vars):
        clean = []
        check = []
        for (j, s, status), t in self.vars.items():
            if [j, s, t] in check:

                if status == "out":
                    clean.remove([int(j), s, "in", t])
                    clean.append([int(j), s, status, t])
            else:
                clean.append([int(j), s, status, t])
                check.append([j, s, t])

        return clean

    def build_timedict(self, clean):

        time_dict = {
            i: {
                "edges": {},
                "stations": {node: [] for node in self.railnet.network.nodes},
            }
            for i in range(self.t_max + 2)
        }

        for j, s, w, t in clean:
            time_dict[t]["stations"][s].append((j, w))

        states = {(j, t): (s, w) for [j, s, w, t] in clean}

        for train_id, train in self.railnet._trains.items():
            for t in range(self.t_max + 2):
                if not (train_id, t) in states:
                    if t > 0:
                        last_station, last_state = states[train_id, t - 1]
                    else:
                        last_station = train.route[0]
                        last_state = "wait"
                        states[train_id, t] = (last_station, last_state)

                    if last_station == train.route[-1]:
                        if last_state == "out":
                            states[train_id, t - 1] = (last_station, "terminated")
                            time_dict[t - 1]["stations"][last_station].remove(
                                (train_id, "out")
                            )
                            time_dict[t - 1]["stations"][last_station].append(
                                (train_id, "terminated")
                            )

                        states[train_id, t] = (last_station, "terminated")
                        time_dict[t]["stations"][last_station].append(
                            (train_id, "terminated")
                        )
                    else:
                        if last_state == "in" or last_state == "wait":
                            states[train_id, t] = (last_station, "wait")

                            time_dict[t]["stations"][last_station].append(
                                (train_id, "wait")
                            )

                        elif last_state == "out" or last_state == "traveling":
                            next_station = None
                            while next_station is None:

                                next_station = train.next_station(last_station)

                            states[train_id, t] = (last_station, "traveling")

                            time_dict[t]["edges"][last_station, next_station] = (
                                train_id,
                                f"to {next_station}",
                            )

        return time_dict

    def plot_solution(self):
        @interact(t=(0, self.t_max))
        def plot_graph(t=0):
            return self.railnet.draw_solution_step(self.time_dict_for_step[t])
        
    def plot_solution_py(self):
    
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.3) 

        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Time', 0, self.t_max, valinit=0, valstep=1)

        self.railnet.draw_solution_step_py(self.time_dict_for_step[0], ax)

        def update(val):
            t = int(val) 
            ax.clear()
            self.railnet.draw_solution_step_py(self.time_dict_for_step[t], ax)

            ax.set_title(f'Solution at time step {t}')
            ax.set_xlabel('X-Achse')
            ax.set_ylabel('Y-Achse')
            ax.autoscale()
            ax.margins(0.15)
            fig.tight_layout()

            fig.canvas.draw_idle() 

        slider.on_changed(update)

        ax.set_title('Solution at time step 0')
        ax.set_xlabel('X-Achse')
        ax.set_ylabel('Y-Achse')

        plt.show()


    def plot(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.3) 
        self.railnet.draw_solution_step_py(self.time_dict_for_step[0], ax)

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bprev = Button(axprev, 'Previous')

        def update_plot():
            ax.clear()
            self.railnet.draw_solution_step_py(self.time_dict_for_step[self.current_t], ax)
            ax.set_title(f'Solution at time step {self.current_t}')
            ax.set_xlabel('X-Achse')
            ax.set_ylabel('Y-Achse')
            ax.autoscale()
            ax.margins(0.15)
            fig.canvas.draw_idle()

        def next_step(event):
            if self.current_t < self.t_max:
                self.current_t += 1
                update_plot()

        def prev_step(event):
            if self.current_t > 0:
                self.current_t -= 1
                update_plot()

        bnext.on_clicked(next_step)
        bprev.on_clicked(prev_step)

        ax.set_title('Solution at time step 0')
        ax.set_xlabel('X-Achse')
        ax.set_ylabel('Y-Achse')

        plt.show()

