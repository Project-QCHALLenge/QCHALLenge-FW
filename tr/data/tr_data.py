from .railnetwork import Train, RailNetwork
from dataclasses import dataclass
from typing import Union
import networkx as nx
import random


@dataclass(order=True)
class TRData:
    EXAMPLES = {
        1: "3-5-slow",
        2: "2-2-opposite",
        3: "2-2-same",
        4: "2-6-dimond",
        5: "2-6-dimond_reverse",
        6: "4-5-X",
        7: "4-5-house",
    }

    def __init__(self, railnet:RailNetwork=None) -> None:

        self.railnet = railnet
        self.num_variables = len(self.railnet.network.nodes) *len(self.railnet._trains)**2

    @classmethod
    def create_problem(cls,
        stations: int = 2,
        trains: int = 2,
        connectivity: float = 0.5,
        seed: int = None,
        min_path_len: int = 1,
        method: str = "gnp",):
        return cls.from_random(stations, trains, connectivity, seed, min_path_len, method)

    def to_dict(self):
        return {
            "railnet": self.railnet.to_dict() if hasattr(self.railnet, "to_dict") else str(self.railnet),
            "num_variables": int(self.num_variables)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TRData":
        railnet = RailNetwork.from_dict(data["railnet"])
        instance = cls(railnet=railnet)
        instance.num_variables = data["num_variables"]
        return instance
        
    def get_num_variables(self):
        return self.num_variables

    def draw(self, **kwargs):
        self.railnet.draw(**kwargs)

    def draw_solution_step(self, **kwargs):
        self.railnet.draw_solution_step(**kwargs)
    
    @classmethod
    def get_example(cls, example: Union[str, int] = 1):
        """
        Return the railnetwork including trains and schedules for either an id or a name of the example
        """

        if example == 1 or example == "3-4-Y":
            G = nx.Graph()
            G.add_node("A", node_type="station", cap=2, label="Station A")
            G.add_node("B", node_type="station", cap=2, label="Station B")
            G.add_node("C", node_type="station", cap=2, label="Station C")
            G.add_node("D", node_type="station", cap=2, label="Station D")
            G.add_node("E", node_type="station", cap=2, label="Station E")

            G.add_edge("A", "C", distance=10, max_speed=10)
            G.add_edge("B", "C", distance=10, max_speed=10)
            G.add_edge("C", "D", distance=10, max_speed=10)
            G.add_edge("D", "E", distance=10, max_speed=10)

            j1 = Train((("A", 0, 1), ("C", 2, 3), ("D", 4, 5), ("E", 6, 7)), priority=1)
            j2 = Train((("E", 0, 1), ("D", 2, 3), ("C", 4, 5), ("A", 6, 7)), priority=1)
            j3 = Train(
                (("B", 0, 1), ("C", 3, 4), ("D", 6, 7), ("E", 9, 10)),
                priority=1,
                speed=5,
            )
            railnet = RailNetwork(G, [j1, j2, j3])

        elif example == 2 or example == "2-2-opposite":
            desc = "Two trains starting on opposite stations. One train is only half as fast and should wait for the faster one to complete its route"
            G = nx.Graph()
            G.add_node("A", node_type="station", cap=2, label="Station A")
            G.add_node("B", node_type="station", cap=2, label="Station B")
            G.add_edge("A", "B", distance=10, max_speed=10)

            j1 = Train((("A", 0, 1), ("B", 2, 3)), priority=1)
            j2 = Train((("B", 0, 1), ("A", 3, 4)), priority=1, speed=5)
            railnet = RailNetwork(G, [j1, j2], desc)

        elif example == 3 or example == "2-2-same":
            desc = "Two trains starting in the same direction. One train is only half as fast and should wait for the faster one to complete its route"
            G = nx.Graph()
            G.add_node("A", node_type="station", cap=2, label="Station A")
            G.add_node("B", node_type="station", cap=2, label="Station B")
            G.add_edge("A", "B", distance=10, max_speed=10)

            j1 = Train((("A", 0, 1), ("B", 2, 3)), priority=1)
            j2 = Train((("A", 0, 1), ("B", 3, 4)), priority=1, speed=5)
            railnet = RailNetwork(G, [j1, j2], desc)

        elif example == 4 or example == "2-6-dimond":
            desc = "Two trains starting in the same direction with multiple stations on their route, diverging and merging routes"
            G = nx.Graph()
            G.add_node("A", node_type="station", cap=2, label="Station A")
            G.add_node("B", node_type="station", cap=2, label="Station B")
            G.add_node("C", node_type="station", cap=2, label="Station C")
            G.add_node("D", node_type="station", cap=2, label="Station D")
            G.add_node("E", node_type="station", cap=2, label="Station E")
            G.add_node("F", node_type="station", cap=2, label="Station F")

            G.add_edge("A", "B", distance=10, max_speed=10)
            G.add_edge("B", "C", distance=10, max_speed=10)
            G.add_edge("B", "D", distance=10, max_speed=10)
            G.add_edge("C", "E", distance=10, max_speed=10)
            G.add_edge("D", "E", distance=10, max_speed=10)
            G.add_edge("E", "F", distance=10, max_speed=10)

            j1 = Train(
                (("A", 0, 1), ("B", 2, 3), ("C", 4, 5), ("E", 6, 7), ("F", 8, 9)),
                priority=1,
            )
            j2 = Train(
                (("A", 0, 1), ("B", 2, 3), ("D", 4, 5), ("E", 6, 7), ("F", 8, 9)),
                priority=1,
            )
            railnet = RailNetwork(G, [j1, j2], desc)

        elif example == 5 or example == "2-6-dimond_reverse":
            desc = "Two trains starting on opposite end with multiple common routes"
            G = nx.Graph()
            G.add_node("A", node_type="station", cap=2, label="Station A")
            G.add_node("B", node_type="station", cap=2, label="Station B")
            G.add_node("C", node_type="station", cap=2, label="Station C")
            G.add_node("D", node_type="station", cap=2, label="Station D")
            G.add_node("E", node_type="station", cap=2, label="Station E")
            G.add_node("F", node_type="station", cap=2, label="Station F")

            G.add_edge("A", "B", distance=10, max_speed=10)
            G.add_edge("B", "C", distance=10, max_speed=10)
            G.add_edge("B", "D", distance=10, max_speed=10)
            G.add_edge("C", "E", distance=10, max_speed=10)
            G.add_edge("D", "E", distance=10, max_speed=10)
            G.add_edge("E", "F", distance=10, max_speed=10)

            j1 = Train(
                (("A", 0, 1), ("B", 2, 3), ("C", 4, 5), ("E", 6, 7), ("F", 8, 9)),
                priority=1,
            )
            j2 = Train(
                (("F", 0, 1), ("E", 2, 3), ("D", 4, 5), ("B", 6, 7), ("A", 8, 9)),
                priority=1,
            )
            railnet = RailNetwork(G, [j1, j2], desc)

        elif example == 6 or example == "4-5-X":
            desc = "4 Trains meeting at a main station."
            G = nx.Graph()
            G.add_node("A", node_type="station", cap=2, label="Station A")
            G.add_node("A1", node_type="station", cap=2, label="Station A")
            G.add_node("B", node_type="station", cap=2, label="Station B")
            G.add_node("C", node_type="station", cap=2, label="Station C")
            G.add_node("D", node_type="station", cap=2, label="Station D")
            G.add_node("E", node_type="station", cap=2, label="Station E")

            G.add_edge("A", "C", distance=10, max_speed=10)
            G.add_edge("A", "A1", distance=10, max_speed=10)
            G.add_edge("B", "C", distance=10, max_speed=10)
            G.add_edge("C", "D", distance=10, max_speed=10)
            G.add_edge("C", "E", distance=10, max_speed=10)

            j1 = Train(
                (("A1", 0, 1), ("A", 2, 3), ("C", 4, 5), ("D", 6, 7)), priority=1
            )
            j2 = Train((("B", 0, 1), ("C", 2, 3), ("D", 4, 5)), priority=1)
            j3 = Train((("B", 0, 1), ("C", 2, 3), ("E", 4, 5)), priority=1)
            j4 = Train((("D", 0, 1), ("C", 2, 3), ("E", 4, 5)), priority=1)
            railnet = RailNetwork(G, [j1, j2, j3, j4])

        elif example == 7 or example == "4-5-house":
            desc = "Two trains starting on opposite end with multiple common routes"
            G = nx.Graph()
            G.add_node("A", node_type="station", cap=2, label="Station A")
            G.add_node("B", node_type="station", cap=2, label="Station B")
            G.add_node("C", node_type="station", cap=2, label="Station C")
            G.add_node("D", node_type="station", cap=2, label="Station D")
            G.add_node("E", node_type="station", cap=2, label="Station E")

            G.add_edge("A", "B", distance=10, max_speed=10)
            G.add_edge("A", "C", distance=10, max_speed=10)
            G.add_edge("B", "C", distance=10, max_speed=10)
            G.add_edge("B", "D", distance=10, max_speed=10)
            G.add_edge("C", "D", distance=10, max_speed=10)
            G.add_edge("C", "E", distance=10, max_speed=10)
            G.add_edge("D", "E", distance=10, max_speed=10)

            j1 = Train(
                ["C", "B", "D", "E"],
                priority=1,
            )
            j2 = Train(
                ["E", "D", "B", "A"],
                priority=1,
            )
            j3 = Train(
                ["B", "D", "C", "E"],
                priority=1,
            )
            j4 = Train(
                ["D", "B", "A", "C"],
                priority=1,
            )
            railnet = RailNetwork(G, [j1, j2, j3, j4], desc)

        else:
            raise ValueError(
                f"Example {example} not found, try one of these:\n {cls.EXAMPLES}"
            )
        
        cls.railnet = railnet
        return cls(railnet)

    def dynamic_test_network(self, stations: int = 2, trains: int = 1) -> RailNetwork:
        """
        builds the worst-case railnet to understand variable scaling
        """
        G = nx.Graph()
        if stations < 2 or trains < 1:
            raise ValueError("Min. stations = 2 and at least one train.")
        for s in range(stations):
            G.add_node(s, node_type="station", label=f"Station {s}")
            if s > 0:
                G.add_edge(s - 1, s, distance=10, max_speed=10)
        trains_list = []
        base_schedule = tuple([(s, s, s) for s in range(stations)])
        for j in range(trains):
            t_schedule = tuple([(s, a + j, b + j) for (s, a, b) in base_schedule])
            trains_list.append(Train(t_schedule))
        self.railnet = RailNetwork(G, trains_list, "worst case railnet")

    @classmethod
    def check_examples(cls):
        return cls.EXAMPLES

    @classmethod
    def _network_gnp(cls, stations, connectivity=0.6, seed=None):
        for _ in range(100):
            G = nx.gnp_random_graph(stations, connectivity, seed)
            connected = nx.is_connected(G)
            if connected:
                break
            else:
                seed = random.randint(0, 100000)
                connectivity *= 1.1

        assert (
            connected
        ), "At least one isolated node. Consider increasing the connectivity."

        mapping = {node: str(node) for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)

        random.seed(seed)
        for u, v in G.edges():
            G[u][v]["distance"] = random.choice([10, 20])
            G[u][v]["max_speed"] = random.choices([10, 5], weights=[80, 20])[0]

        railnet = RailNetwork(
            G, desc={"seed": seed, "stations": stations, "connectivity": connectivity}
        )
        return railnet

    def _network_prim(self, stations, seed):

        G = nx.Graph()
        nodes = list(range(stations))
        random.seed(seed)
        random.shuffle(nodes)

        G.add_node(nodes.pop())

        while nodes:

            new_node = nodes.pop()
            existing_node = random.choice(list(G.nodes))
            G.add_edge(existing_node, new_node)
        for u, v in G.edges():
            G[u][v]["distance"] = random.choice([10, 20])
            G[u][v]["max_speed"] = random.choices([10, 5], weights=[80, 20])[0]
        railnet = RailNetwork(G, desc={"seed": seed, "stations": stations})
        return railnet

    def _network_bag(self, stations, connectivity, seed):

        G = nx.barabasi_albert_graph(stations, connectivity, seed)
        for u, v in G.edges():
            G[u][v]["distance"] = random.choice([10, 20])
            G[u][v]["max_speed"] = random.choices([10, 5], weights=[80, 20])[0]

        railnet = RailNetwork(
            G, desc={"seed": seed, "stations": stations, "connectivity": connectivity}
        )
        return railnet

    @classmethod
    def from_random(
        cls,
        stations: int = 2,
        trains: int = 2,
        connectivity: float = 0.5,
        seed: int = None,
        min_path_len: int = 1,
        method: str = "gnp",
    ):

        if seed is None:
            seed = random.randint(0, 100000)
        else:
            random.seed(seed)
        if method == "gnp":
            railnet = cls._network_gnp(
                stations=stations, connectivity=connectivity, seed=seed
            )
        elif method == "prim":
            railnet = cls._network_prim(stations=stations, seed=seed)

        for j in range(trains):
            random.seed(seed + j)
            railnet.add_random_train(seed=seed + j, min_path_length=min_path_len)

        cls.railnet = railnet
        return cls(railnet)

    @classmethod
    def periodic_network(cls, repeats: int = 1):
        G = nx.Graph()
        G.add_node("MW", node_type="station", cap=2, label="Max Weber")
        G.add_node("OP", node_type="station", cap=2, label="Odeons Pl")
        G.add_node("KP", node_type="station", cap=2, label="Kolumbus Pl")
        G.add_node("IS", node_type="station", cap=2, label="Impler Str")
        G.add_node("ST", node_type="station", cap=2, label="Sendlinger T")
        G.add_node("HBF", node_type="station", cap=2, label="HBF")
        G.add_node("MF", node_type="station", cap=2, label="Münchener Fr")
        G.add_node("U4U5west", node_type="station", cap=2, label="U4 U5 West")
        G.add_node("U4east", node_type="station", cap=2, label="U4 East")
        G.add_node("U5east", node_type="station", cap=2, label="U5 East")
        G.add_node("U1west", node_type="station", cap=2, label="U1 West")
        G.add_node("U1east", node_type="station", cap=2, label="U1 East")
        G.add_node("U2west", node_type="station", cap=2, label="U2 West")
        G.add_node("U2east", node_type="station", cap=2, label="U2 East")
        G.add_node("U3south", node_type="station", cap=2, label="U3 South")
        G.add_node("U3north", node_type="station", cap=2, label="U3 North")
        G.add_node("U6south", node_type="station", cap=2, label="U6 South")
        G.add_node("U6north", node_type="station", cap=2, label="U6 North")

        G.add_edge("MW", "U4east")
        G.add_edge("MW", "U5east")
        G.add_edge("MW", "OP")
        G.add_edge("MF", "OP")
        G.add_edge("ST", "OP")
        G.add_edge("HBF", "OP")
        G.add_edge("MF", "U6north")
        G.add_edge("MF", "U3north")
        G.add_edge("HBF", "U1west")
        G.add_edge("HBF", "U2west")
        G.add_edge("HBF", "U4U5west")
        G.add_edge("HBF", "ST")
        G.add_edge("IS", "ST")
        G.add_edge("KP", "ST")
        G.add_edge("KP", "U1east")
        G.add_edge("KP", "U2east")
        G.add_edge("IS", "U6south")
        G.add_edge("IS", "U3south")

        for u, v in G.edges():
            G[u][v]["distance"] = 10
            G[u][v]["max_speed"] = 10
        railnet = RailNetwork(G, desc={"periodic": repeats})

        u1_west_route = ["U1west", "HBF", "ST", "KP", "U1east"]
        u2_west_route = ["U2west", "HBF", "ST", "KP", "U2east"]

        u1_east_route = u1_west_route[::-1]
        u2_east_route = u2_west_route[::-1]

        u4_west_route = ["U4U5west", "HBF", "OP", "MW", "U4east"]
        u5_west_route = ["U4U5west", "HBF", "OP", "MW", "U5east"]

        u4_east_route = u4_west_route[::-1]
        u5_east_route = u5_west_route[::-1]

        u3_north_route = ["U3north", "MF", "OP", "ST", "IS", "U3south"]
        u6_north_route = ["U6north", "MF", "OP", "ST", "IS", "U6south"]

        u3_south_route = u3_north_route[::-1]
        u6_south_route = u6_north_route[::-1]

        for r in range(repeats):

            railnet.add_train(
                Train(
                    tuple(
                        (
                            s,
                            (idx * 2 - 1) - 1 + 2 * (r - 1),
                            (idx * 2 - 1) + 2 * (r - 1),
                        )
                        for idx, s in enumerate(u1_east_route, start=1)
                    )
                )
            )
            railnet.add_train(
                Train(
                    tuple(
                        (s, (idx * 2) - 1 + 2 * (r - 1), (idx * 2) + 2 * (r - 1))
                        for idx, s in enumerate(u2_east_route, start=1)
                    )
                )
            )

            railnet.add_train(
                Train(
                    tuple(
                        (
                            s,
                            (idx * 2 - 1) - 1 + 2 * (r - 1),
                            (idx * 2 - 1) + 2 * (r - 1),
                        )
                        for idx, s in enumerate(u1_west_route, start=1)
                    )
                )
            )
            railnet.add_train(
                Train(
                    tuple(
                        (s, (idx * 2) - 1 + 2 * (r - 1), (idx * 2) + 2 * (r - 1))
                        for idx, s in enumerate(u2_west_route, start=1)
                    )
                )
            )

            railnet.add_train(
                Train(
                    tuple(
                        (
                            s,
                            (idx * 2 - 1) - 1 + 2 * (r - 1),
                            (idx * 2 - 1) + 2 * (r - 1),
                        )
                        for idx, s in enumerate(u4_east_route, start=1)
                    )
                )
            )
            railnet.add_train(
                Train(
                    tuple(
                        (s, (idx * 2) - 1 + 2 * (r - 1), (idx * 2) + 2 * (r - 1))
                        for idx, s in enumerate(u5_east_route, start=1)
                    )
                )
            )

            railnet.add_train(
                Train(
                    tuple(
                        (
                            s,
                            (idx * 2 - 1) - 1 + 2 * (r - 1),
                            (idx * 2 - 1) + 2 * (r - 1),
                        )
                        for idx, s in enumerate(u4_west_route, start=1)
                    )
                )
            )
            railnet.add_train(
                Train(
                    tuple(
                        (s, (idx * 2) - 1 + 2 * (r - 1), (idx * 2) + 2 * (r - 1))
                        for idx, s in enumerate(u5_west_route, start=1)
                    )
                )
            )

            railnet.add_train(
                Train(
                    tuple(
                        (
                            s,
                            (idx * 2 - 1) - 1 + 2 * (r - 1),
                            (idx * 2 - 1) + 2 * (r - 1),
                        )
                        for idx, s in enumerate(u3_south_route, start=1)
                    )
                )
            )
            railnet.add_train(
                Train(
                    tuple(
                        (s, (idx * 2) - 1 + 2 * (r - 1), (idx * 2) + 2 * (r - 1))
                        for idx, s in enumerate(u6_south_route, start=1)
                    )
                )
            )

            railnet.add_train(
                Train(
                    tuple(
                        (
                            s,
                            (idx * 2 - 1) - 1 + 2 * (r - 1),
                            (idx * 2 - 1) + 2 * (r - 1),
                        )
                        for idx, s in enumerate(u3_north_route, start=1)
                    )
                )
            )
            railnet.add_train(
                Train(
                    tuple(
                        (s, (idx * 2) - 1 + 2 * (r - 1), (idx * 2) + 2 * (r - 1))
                        for idx, s in enumerate(u6_north_route, start=1)
                    )
                )
            )

        cls.railnet = railnet
        return cls()
