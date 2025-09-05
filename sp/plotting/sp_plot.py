import matplotlib.pyplot as plt   
import networkx as nx
import matplotlib.cm as cm
from abstract.plot.abstract_plot import AbstractPlot


class SPPlot(AbstractPlot):
     def __init__(self, evaluation = None, data = None):
          if data:
               self.data = data
               self.evaluation = None
          else:
               self.data = evaluation.data
               self.evaluation = evaluation

     def plot_solution(self, hide_never_covered = True):
          pos = {node: node for node in self.data.G.nodes()}
          pos_opt= {node: node for node in self.evaluation.O.nodes()}

          if hide_never_covered: 
               street_points_covered = set(self.data.listStreetPoints) - set(self.data.listStreetPointsNeverCovered)
               nx.draw_networkx_nodes(self.data.G, pos, nodelist=list(street_points_covered), node_color='blue', node_size=40)
          else: 
               nx.draw_networkx_nodes(self.data.G, pos, self.data.listStreetPoints, node_color = 'blue', node_size= 40)
               nx.draw_networkx_nodes(self.data.G, pos, self.data.listStreetPointsNeverCovered, node_color = 'red', node_size= 10)

          nx.draw_networkx_nodes(self.evaluation.O, pos_opt, self.evaluation.listStreetPointsCovered, node_color = 'green', node_size= 40)
          nx.draw_networkx_nodes(self.data.G, pos, self.data.listLidar, node_color = 'orange', node_size= 40)
          nx.draw_networkx_nodes(self.evaluation.O, pos_opt, self.evaluation.listLidarActivated, node_color = 'red', node_size= 40)
          
          posw = {node: node for node in self.data.M.nodes()}
          edw = {node: node for node in self.data.M.edges()}
          
          nx.draw_networkx_edges(self.data.M,posw, edw, width=5.0, alpha=1, edge_color='black')

          plt.axis('equal')
          plt.grid(True)
          
          plt.title("Solution")
          plt.text(0.5, -0.05, f'Activated Lidars: {self.evaluation.get_objective()} \n missing achievable coverage: {self.evaluation.check_solution()["missing_achievable_coverage"]}', fontsize=10, ha='center', va='top', transform=plt.gca().transAxes)
          plt.plot
          return plt


     def plot_problem(self, draw_connections = True, hide_never_covered = True):
          pos = {node: node for node in self.data.G.nodes()}

          if hide_never_covered: 
               street_points_covered = set(self.data.listStreetPoints) - set(self.data.listStreetPointsNeverCovered)
               nx.draw_networkx_nodes(self.data.G, pos, nodelist=list(street_points_covered), node_color='blue', node_size=40)
          else: 
               nx.draw_networkx_nodes(self.data.G, pos, self.data.listStreetPoints, node_color = 'blue', node_size= 40)
               nx.draw_networkx_nodes(self.data.G, pos, self.data.listStreetPointsNeverCovered, node_color = 'red', node_size= 10)
          nx.draw_networkx_nodes(self.data.G, pos, self.data.listLidar, node_color = 'orange', node_size= 40)

          posw = {node: node for node in self.data.M.nodes()}
          edw = {node: node for node in self.data.M.edges()}
          nx.draw_networkx_edges(self.data.M,posw, edw, width=5.0, alpha=1, edge_color='black')

          if draw_connections: 
               posl = { (node[0], node[1]): (node[0], node[1]) for node in self.data.G.nodes()}
               edl = {  ((node[0][0], node[0][1]), (node[1][0], node[1][1])): (node[0], node[1]) for node in self.data.G.edges()}
               nx.draw_networkx_edges(self.data.G,posl, edl, width=2.0, alpha=0.5, edge_color='green')

          plt.axis('equal')
          plt.grid(True)

          return plt

     def plot_reduced_problem(self, draw_connections = True):
          pos = {node: node for node in self.data.G.nodes()}
          removed_lidar = self.data.lidar0 + self.data.lidar1

          street_points_covered = set(self.data.listStreetPoints) - set(self.data.listStreetPointsNeverCovered)

          nx.draw_networkx_nodes(self.data.G, pos, nodelist=list(street_points_covered), node_color='blue', node_size=40)
          nx.draw_networkx_nodes(self.data.G, pos, self.data.listLidar, node_color = 'orange', node_size= 40)
          nx.draw_networkx_nodes(self.data.G, pos, removed_lidar, node_color = 'red', node_size= 40)

          posw = {node: node for node in self.data.M.nodes()}
          edw = {node: node for node in self.data.M.edges()}
          nx.draw_networkx_edges(self.data.M,posw, edw, width=5.0, alpha=1, edge_color='black')

          if draw_connections:
               posl = { (node[0], node[1]): (node[0], node[1]) for node in self.data.G.nodes()}
               edl = {  ((node[0][0], node[0][1]), (node[1][0], node[1][1])): (node[0], node[1]) for node in self.data.G_reduced.edges()}
               nx.draw_networkx_edges(self.data.G,posl, edl, width=2.0, alpha=0.5, edge_color='green')

               reduced_edges = self.data.G.edges() - self.data.G_reduced.edges()
               edl = {  ((node[0][0], node[0][1]), (node[1][0], node[1][1])): (node[0], node[1]) for node in reduced_edges}
               nx.draw_networkx_edges(self.data.G, posl, edl, width=2.0, alpha=0.5, edge_color='red')

          plt.axis('equal')
          plt.grid(True)

          return plt
     
     def plot_subproblems(self, decomposer):
          fig, ax = plt.subplots()
          G = self.data.G

          subproblems = decomposer.subproblems
          num_subproblems = len(subproblems)

          if num_subproblems <= 10:
               cmap = cm.get_cmap("tab10")
               colors = [cmap(i) for i in range(num_subproblems)]
          else:
               cmap = cm.get_cmap("viridis", num_subproblems)
               colors = [cmap(i) for i in range(num_subproblems)]

          pos = {node: (node[0], node[1]) for node in G.nodes()}
          for i, subproblem in enumerate(subproblems):
               nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=list(subproblem.listLidar),
                    node_color=[colors[i]] * len(subproblem.listLidar),
                    node_size=50,
                    label=f"Lidar Partition {i+1}",
                    edgecolors='black',
                    linewidths=2,
                    ax=ax
               )
               street_points_covered = set(subproblem.listStreetPoints) - set(self.data.listStreetPointsNeverCovered)
               nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=list(street_points_covered),
                    node_color=[colors[i]] * len(street_points_covered),
                    node_size=50,
                    label=f"Streetpoints Partition {i+1}",
                    ax=ax
               )
               subgraph = G.subgraph(subproblem.G)
               nx.draw_networkx_edges(
                    G, 
                    pos, 
                    nodelist=list(subgraph), 
                    edgelist=subgraph.edges(), 
                    edge_color=colors[i], 
                    alpha=0.7, 
                    width=2.0, 
                    ax=ax
               )

          plt.axis("equal")
          plt.grid(True)
          plt.legend(loc="best")
          ax.set_title("Partitioned Graph with Minimal Cuts")

          return plt
     
     def plot_partitioned_graph(self, partitions, cut_edges, plot_cut_edges = True):
        """
        Plot the graph with partitions highlighted.

        Parameters:
        - partitions: List of sets, each containing nodes in a partition.
        - cut_edges: List of edges that are cut (inter-partition edges).

        Returns:
        - fig: The matplotlib figure object with the plot.
        """
        fig, ax = plt.subplots()
        G = self.data.G

        # Generate positions ensuring consistent dimensionality
        pos = {node: (node[0], node[1]) for node in G.nodes()}  # Use (x, y) for all nodes

        # Assign unique colors to each partition
        if len(partitions) <= 10:
            cmap = cm.get_cmap("tab10")
            colors = [cmap(i) for i in range(len(partitions))]
        else:
            cmap = cm.get_cmap("viridis", len(partitions))
            colors = [cmap(i) for i in range(len(partitions))]

        # Plot partitions
        for i, part in enumerate(partitions):
            covered_part = set(part) - set(self.data.listStreetPointsNeverCovered)
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=list(covered_part),
                node_color=[colors[i]] * len(covered_part),
                node_size=50,
                label=f"Partition {i+1}",
                ax=ax
            )

        # Plot intra-partition edges
        for i, part in enumerate(partitions):
            subgraph = G.subgraph(part)
            nx.draw_networkx_edges(G, pos, nodelist=list(subgraph), edgelist=subgraph.edges(), edge_color=colors[i], alpha=0.7, width=2.0, ax=ax)

        # Plot cut edges (inter-partition edges)
        if plot_cut_edges:
            nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color="red", alpha=0.9, width=2.5, label="Cut Edges", ax=ax)

        # Add grid, legend, and title
        ax.axis("equal")
        ax.grid(True)
        ax.legend(loc="best")
        ax.set_title("Partitioned Graph with Minimal Cuts")

        return fig

