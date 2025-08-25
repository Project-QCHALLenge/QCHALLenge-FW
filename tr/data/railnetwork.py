import networkx as nx
import matplotlib.pyplot as plt
from typing import Union
import random
from matplotlib import patheffects


class InvalidScheduleError(Exception):
    pass


class Train:
    id = None
    route: list = None
    color = None

    def __init__(
        self,
        schedule: Union[
            list, tuple
        ],  # list assumes a route ["A","B"...]; tuple assumes a schedule ((A,0,1),(B,2,3)...)
        speed: int = 10,
        priority=1,
    ):
        if isinstance(schedule, tuple):
            self.schedule = schedule  # schedule = ((A,0,1),(C,2,3),(D,4,5))  nodes and their arival and departure times
            self.route = list(s[0] for s in schedule)

        elif isinstance(schedule, list):
            self.route = schedule
            self.schedule = None
        else:
            raise ValueError("Wrong schedule type.")
        self.speed = speed
        self.priority = priority

    def next_station(self, station):

        index = self.route.index(station)

        if index < len(self.route) - 1:
            return self.route[index + 1]
        else:
            None

    def previous_station(self, station):

        index = self.route.index(station)

        if index > 0:
            return self.route[index - 1]
        else:
            None


class RailNetwork:
    """
    class properties:
    graph G(V,E)  Vertices: Stations (with ?capacity?),
            Edges: Lines (with max speeds, length, ?capacity or track-type(single, double, multi)?)
    J trains: with speeds, priority, route(tuple of stations), schedule(departure times)
    C contraints to take acount for

    """

    def __init__(
        self,
        G: nx.graph = None,
        trains=None,
        desc=None,
    ) -> None:
        self.network: nx.graph = G
        self._pos = nx.spring_layout(G)
        self._trains = {}
        self.next_trainID = 1
        self.description = desc
        self.dmax = 2
        if trains is not None:
            if isinstance(trains, list):
                for j in trains:
                    self.add_train(j)

            elif isinstance(trains, Train):
                self.add_train(j)

    def to_dict(self):
        return {
            "nodes": list(self.network.nodes),
            "edges": list(self.network.edges(data=True)),  # includes attributes
            "positions": {k: v.tolist() for k, v in self._pos.items()},
            "trains": {
                train_id: train.to_dict() if hasattr(train, "to_dict") else str(train)
                for train_id, train in self._trains.items()
            },
            "next_trainID": self.next_trainID,
            "description": self.description,
            "dmax": self.dmax,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RailNetwork":
        G = nx.DiGraph()
        for node in data["nodes"]:
            G.add_node(str(node))

        for edge_str, attr in data["edges"].items():
            edge = eval(edge_str)  # turns "('0', '1')" → ('0', '1')
            G.add_edge(str(edge[0]), str(edge[1]), **attr)

        trains = []
        for train_id, train_data in data["trains"].items():
            weight, capacity, path = train_data
            trains.append(Train(train_id=int(train_id), weight=weight, capacity=capacity, path=path))

        instance = cls(G=G, trains=trains)
        return instance

    def __str__(self) -> str:
        __trains = [
            f"\n      ID: {k}, schedule: {t.schedule}" for k, t in self._trains.items()
        ]
        str_trains = ""
        for st in __trains:
            str_trains += st
        return f"Railnetwork\n   Nodes:\n      {self.network.nodes}\n   Trains:{str_trains}"

    def draw_solution_step_py(self, time_dict_for_step: dict, ax):
        # Die Logik deines Netzwerksplots bleibt gleich, aber wir zeichnen explizit auf `ax`
        min_y = min(self._pos[node][1] for node in self._pos)
        max_y = max(self._pos[node][1] for node in self._pos)
        plot_height = max(0.1, max_y - min_y)
        edge_dict = time_dict_for_step["edges"]
        station_dict = time_dict_for_step["stations"]

        station_labels = {node: f"{node}" for node in self.network.nodes}
        for node in station_labels.keys():
            if len(station_dict[node]) > 0:
                x, y = self._pos[node]
                y -= 0.03 * plot_height
                for train, status in station_dict[node]:
                    y -= 0.04 * plot_height
                    text = ax.text(  # Hier verwenden wir `ax.text` statt `plt.text`
                        x,
                        y,
                        f"{train} : {status}",
                        color=self._trains[train].color,
                    )
                    text.set_path_effects(
                        [
                            patheffects.Stroke(linewidth=1, foreground="gray"),
                            patheffects.Normal(),
                        ]
                    )

        station_pos = {
            k: (v[0], v[1] + plot_height * 0.02) for k, v in self._pos.items()
        }

        if len(edge_dict) > 0:
            edge_labels = {
                (u, v): [f"{train}: {status}", train]
                for (u, v), (train, status) in edge_dict.items()
            }
        else:
            edge_labels = {}

        # Zeichnen der Netzwerkelemente auf `ax`, nicht auf `plt.gca()`
        nx.draw_networkx_nodes(
            self.network,
            self._pos,
            ax=ax  # Verwende `ax` explizit für das Zeichnen
        )

        for edge, label in edge_labels.items():
            edge_text = nx.draw_networkx_edge_labels(
                self.network,
                self._pos,
                edge_labels={edge: label[0]},
                label_pos=0.5,
                font_color=self._trains[label[1]].color,
                ax=ax  # Verwende `ax` explizit
            )
            for _, t in edge_text.items():
                t.set_path_effects(
                    [
                        patheffects.Stroke(linewidth=1, foreground="gray"),
                        patheffects.Normal(),
                    ]
                )

        nx.draw_networkx_labels(
            self.network,
            station_pos,
            station_labels,
            horizontalalignment="center",
            verticalalignment="top",
            ax=ax  # Verwende `ax` explizit
        )
        nx.draw_networkx_edges(
            self.network,
            self._pos,
            ax=ax  # Verwende `ax` explizit
        )

        # Skaliere die Achse automatisch
        ax.autoscale()
        ax.margins(0.15)

    def draw_solution_step(self, time_dict_for_step: dict):

        min_y = min(self._pos[node][1] for node in self._pos)
        max_y = max(self._pos[node][1] for node in self._pos)
        plot_height = max(0.1, max_y - min_y)
        edge_dict = time_dict_for_step["edges"]
        station_dict = time_dict_for_step["stations"]

        station_labels = {node: f"{node}" for node in self.network.nodes}
        for node in station_labels.keys():
            if len(station_dict[node]) > 0:
                x, y = self._pos[node]
                y -= 0.03 * plot_height
                for train, status in station_dict[node]:

                    y -= 0.04 * plot_height
                    text = plt.text(
                        x,
                        y,
                        f"{train} : {status}",
                        color=self._trains[train].color,
                    )
                    text.set_path_effects(
                        [
                            patheffects.Stroke(linewidth=1, foreground="gray"),
                            patheffects.Normal(),
                        ]
                    )

        station_pos = {
            k: (v[0], v[1] + plot_height * 0.02) for k, v in self._pos.items()
        }

        if len(edge_dict) > 0:
            edge_labels = {
                (u, v): [f"{train}: {status}", train]
                for (u, v), (train, status) in edge_dict.items()
            }
        else:
            edge_labels = {}

        nx.draw_networkx_nodes(
            self.network,
            self._pos,
        )

        for edge, label in edge_labels.items():

            edge_text = nx.draw_networkx_edge_labels(
                self.network,
                self._pos,
                edge_labels={edge: label[0]},
                label_pos=0.5,
                font_color=self._trains[label[1]].color,
            )
            for _, t in edge_text.items():
                t.set_path_effects(
                    [
                        patheffects.Stroke(linewidth=1, foreground="gray"),
                        patheffects.Normal(),
                    ]
                )

        nx.draw_networkx_labels(
            self.network,
            station_pos,
            station_labels,
            horizontalalignment="center",
            verticalalignment="top",
        )
        nx.draw_networkx_edges(
            self.network,
            self._pos,
        )

        # nx.draw_networkx_edge_labels(self.network, self._pos, edge_labels=edge_labels)

        plt.gca().autoscale()
        plt.gca().margins(0.15)
        plt.tight_layout()

    def draw(self, details=True):
        node_labels = {node: f"{node}" for node in self.network.nodes}

        for t in self._trains.values():
            start_node = t.schedule[0][0]
            if details:
                node_labels[start_node] += f"\n {t.id} : {t.speed},{t.route}"
            else:
                node_labels[start_node] += f"\n {t.id}"

        if details:
            edge_labels = {
                (u, v): f"s: {data['distance']}\nv: {data['max_speed']}"
                for u, v, data in self.network.edges(data=True)
            }
        else:
            edge_labels = {(u, v): " " for u, v, data in self.network.edges(data=True)}

        # Draw edge labels
        nx.draw_networkx_edge_labels(
            self.network,
            pos=self._pos,
            edge_labels=edge_labels,
            label_pos=0.5,
            font_color="black",
            font_size=8,
        )

        nx.draw(
            self.network,
            pos=self._pos,
            with_labels=True,
            labels=node_labels,
            font_size=8,
            horizontalalignment="left",
            verticalalignment="top",
        )
        plt.gca().autoscale()
        plt.gca().margins(0.15)
        plt.tight_layout()
        plt.show()

    def _check_validity(self, schedule):
        # schedule is valid iff:
        # possible for the train to reach(!) all stations in time(!), given no additional delay
        # contains no loops

        def _contains_loops():
            # to implement
            pass

        def _slow_train():
            # to implement
            pass

        def _start_stop():
            return len(schedule) < 2

        def _route():
            # to implement
            pass

        if _route():
            raise InvalidScheduleError("Stations are not connected.")

        if _start_stop():
            raise InvalidScheduleError("Needs at least a start and end station!")

        if _contains_loops():
            raise InvalidScheduleError("Contains loops!")

        if _slow_train():
            raise InvalidScheduleError(
                "Train can not reach the stations in time. Disconnected network or train too slow!"
            )

    def add_train(self, train: Train):
        """
        adds a train (or list of trains) with their respective schedules to the network
        should also check for conflicts and for the validity of the route
        (only using existing connections, departure times possible)
        """

        self.dmax += 2

        black: str = "black"
        red: str = "red"
        green: str = "green"
        yellow: str = "yellow"
        blue: str = "blue"
        purple: str = "purple"
        cyan = "cyan"
        magenta = "magenta"
        brown = "brown"
        orange = "orange"
        pink = "pink"
        gray = "gray"
        lightblue = "lightblue"
        navy = "navy"
        teal = "teal"
        lime = "lime"
        olive = "olive"
        maroon = "maroon"

        colors = [
            black,
            red,
            green,
            yellow,
            blue,
            purple,
            cyan,
            magenta,
            brown,
            orange,
            pink,
            gray,
            lightblue,
            navy,
            teal,
            lime,
            olive,
            maroon,
        ]

        if train.schedule is None:
            if train.route is None:
                raise ("specify route or schedule")
            train.schedule = self._make_schedule(train.route, train.speed)

        try:
            self._check_validity(train.schedule)
        except InvalidScheduleError as e:
            print(f"Schedule is not valid: {e}")

        if self.next_trainID not in self._trains.keys():
            train.id = self.next_trainID
            train.color = colors[train.id % len(colors)]

            self._trains[self.next_trainID] = train
            self.next_trainID += 1
        else:
            self.next_trainID += 1
            self.add_train(train)

    def _get_subsequents(self, tup):
        """
        from a route ("A","B","C") create subsequent pairs-> (("A","B"),("B","C"))
        """
        pairs = list()
        for i in range(len(tup) - 1):
            pairs.append((tup[i], tup[i + 1]))
        return tuple(pairs)

    def _make_schedule(
        self,
        path: list,
        train_speed: float,
    ):
        subsequents = self._get_subsequents(path)
        passing_times = {}
        for s, s_next in subsequents:
            edge_attributes = self.network.get_edge_data(s, s_next)

            distance = edge_attributes["distance"]
            speed = min(edge_attributes["max_speed"], train_speed)
            passing_times[(s, s_next)] = round(distance / speed)

        schedule = []
        for idx, node in enumerate(path):
            if idx == 0:
                schedule.append((node, 0, 1))
            else:
                prev = path[idx - 1]
                t_in = schedule[idx - 1][2] + passing_times[prev, node]
                schedule.append((node, t_in, t_in + 1))

        schedule = tuple(schedule)
        return schedule

    def add_random_train(self, min_path_length=1, seed=None):

        if seed is None:
            random.seed(random.randint(0, 100000))
        else:
            random.seed(seed)

        nodes = list(self.network.nodes)
        distance_paths = []
        tries = 0
        while len(distance_paths) < 1:
            tries += 1
            if tries > 50:
                raise (
                    "Unable to find path with set path length. Try shorter min path length."
                )
            start_node, end_node = random.sample(nodes, 2)
            all_paths = list(nx.all_simple_paths(self.network, start_node, end_node))
            distance_paths = [
                path for path in all_paths if len(path) >= min_path_length
            ]

        path = random.choices(distance_paths)[0]

        train_speed = random.choices([10, 5], weights=[80, 20])[0]
        schedule = self._make_schedule(path, train_speed)

        self.add_train(Train(schedule=schedule, speed=train_speed))

    def remove_train(self, train_id):
        if train_id in self._trains.keys():
            self.next_trainID = train_id
            del self._trains[self.next_trainID]
        else:
            raise KeyError("Train does not exist in this network!")

    def add_delay(self, train_id, station, delay):
        if train_id not in self._trains:
            raise ValueError(
                f"Train does not exists! Existing trains: {self._trains.keys()}"
            )
        if station not in self._trains[train_id].route:
            raise ValueError(
                f"No valid station for the train. Valid stations: {self._trains[train_id].route}"
            )
        # TODO propagate delay through the remaining route

    def to_dict(self) -> dict:
        data = {"nodes": {}, "edges": {}, "trains": {}}
        for node in self.network.nodes:
            data["nodes"][node] = self.network.nodes[node]

        for edge in self.network.edges:
            data["edges"][str(edge)] = self.network.edges[edge]

        for train in self._trains:
            data["trains"][train] = [
                self._trains[train].speed,
                self._trains[train].priority,
                self._trains[train].route,
            ]
        return data

    @classmethod
    def from_dict(cls, loaded_data: dict):

        G = nx.Graph()
        for node, node_data in loaded_data["nodes"].items():
            G.add_node(node, **node_data)

        edges = {eval(k): v for k, v in loaded_data["edges"].items()}
        for (s, e), properties in edges.items():
            G.add_edge(s, e, **properties)

        railnet = cls(G)

        for id, (speed, prio, route) in loaded_data["trains"].items():
            train = Train(route, speed, prio)
            railnet.add_train(train)

        return railnet
