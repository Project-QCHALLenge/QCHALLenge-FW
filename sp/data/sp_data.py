import json
import math
import numpy as np
import networkx as nx

from pathlib import Path

from .glb_reader_small import create_problem_from_glb
from abstract.data.abstract_data import AbstractData


class SPData(AbstractData):

    def __init__(
            self, 
            max_vert_angle = 16, 
            min_vert_angle = -16, 
            max_radius = 30, 
            half_horizontal_angle = 180, 
            lidar_wall_offset = 0.2
        ):

        self.max_vert_angle = max_vert_angle
        self.min_vert_angle = min_vert_angle
        self.max_radius = max_radius
        self.half_horizontal_angle = half_horizontal_angle 
        self.lidar_wall_offset = lidar_wall_offset

        self.data_params = {
            "max_vert_angle": max_vert_angle,
            "min_vert_angle": min_vert_angle,
            "max_radius": max_radius,
            "half_horizontal_angle": half_horizontal_angle,
            "lidar_wall_offset": lidar_wall_offset
        }

        self.G = nx.Graph()
        self.G_reduced = nx.Graph()
        self.M = nx.Graph()
        
        self.problem_dict = dict()
        self.walls = []
        self.listLidar = []
        self.listStreetPoints = []

        self.listStreetPointsNeverCovered = []

        #undefined
        self.missing_achievable_coverage=None
        self.never_covered=None

        # lidar that covers only streetpoints already covered by single lidar
        self.lidar0 = []
        # lidar that cover SP that is not covered by another lidar
        self.lidar1 = []

    @classmethod
    def create_problem(cls, num_cols = 5, version = 3, max_radius= 2.5, hor_basic_distance = 1, vert_basic_dist = 2, seed = 1):
        return cls.gen_problem(num_cols, version, max_radius, hor_basic_distance, vert_basic_dist)

    def to_dict(self):
        def convert(value):
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (set, tuple)):
                return list(value)
            elif isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert(v) for v in value]
            elif isinstance(value, nx.Graph):
                return {
                    "nodes": list(value.nodes),
                    "edges": list(value.edges)
                }
            else:
                return value

        return {
            "max_vert_angle": self.max_vert_angle,
            "min_vert_angle": self.min_vert_angle,
            "max_radius": self.max_radius,
            "half_horizontal_angle": self.half_horizontal_angle,
            "lidar_wall_offset": self.lidar_wall_offset,
            "data_params": convert(self.data_params),
            "G": convert(self.G),
            "G_reduced": convert(self.G_reduced),
            "M": convert(self.M),
            "problem_dict": convert(self.problem_dict),
            "walls": convert(self.walls),
            "listLidar": convert(self.listLidar),
            "listStreetPoints": convert(self.listStreetPoints),
            "listStreetPointsNeverCovered": convert(self.listStreetPointsNeverCovered),
            "missing_achievable_coverage": self.missing_achievable_coverage,
            "never_covered": self.never_covered,
            "lidar0": convert(self.lidar0),
            "lidar1": convert(self.lidar1),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SPData":
        def convert_nodes(nodes):
            return [tuple(n) if isinstance(n, list) else n for n in nodes]

        def convert_edges(edges):
            return [(tuple(edge[0]), tuple(edge[1])) for edge in edges]

        def safe_tuple_list(lst):
            return [tuple(x) if isinstance(x, list) else x for x in lst]

        instance = cls(
            max_vert_angle=data["max_vert_angle"],
            min_vert_angle=data["min_vert_angle"],
            max_radius=data["max_radius"],
            half_horizontal_angle=data["half_horizontal_angle"],
            lidar_wall_offset=data["lidar_wall_offset"]
        )

        instance.data_params = data["data_params"]
        instance.problem_dict = data["problem_dict"]
        instance.walls = data["walls"]
        instance.listLidar = safe_tuple_list(data["listLidar"])
        instance.listStreetPoints = safe_tuple_list(data["listStreetPoints"])
        instance.listStreetPointsNeverCovered = safe_tuple_list(data["listStreetPointsNeverCovered"])
        instance.lidar0 = safe_tuple_list(data["lidar0"])
        instance.lidar1 = safe_tuple_list(data["lidar1"])
        instance.missing_achievable_coverage = data["missing_achievable_coverage"]
        instance.never_covered = data["never_covered"]

        instance.G = nx.Graph()
        instance.G.add_nodes_from(convert_nodes(data["G"]["nodes"]))
        instance.G.add_edges_from(convert_edges(data["G"]["edges"]))

        instance.G_reduced = nx.Graph()
        instance.G_reduced.add_nodes_from(convert_nodes(data["G_reduced"]["nodes"]))
        instance.G_reduced.add_edges_from(convert_edges(data["G_reduced"]["edges"]))

        instance.M = nx.Graph()
        instance.M.add_nodes_from(convert_nodes(data["M"]["nodes"]))
        instance.M.add_edges_from(convert_edges(data["M"]["edges"]))

        return instance

    def preprocessing(self):
        self.G_reduced = self.G.copy()
        self.find_similar_lidar()
        self.remove_slack_zero()

    def remove_slack_zero(self):
        """
        This function removes lidars if a street point is connected to this unique lidar.
        """
        to_delete = []
        for node in self.G_reduced.nodes:
            if node not in to_delete:
                # If the node is a "street point"
                if len(node) == 3:
                    # Check if the node has neighbors
                    if len(self.G_reduced.adj[node]) == 0:
                        pass
                        # raise ValueError(f"Isolated node detected: {node}. This node has no neighbors.")
                        # print(f"Isolated node detected: {node}. This node has no neighbors.")
                    else:
                        if len(self.G_reduced.adj[node]) == 1: #If len == 1 we know that v_i is connected to a unique lidar (which can then be trivially put to 1)
                            # Get the first lidar
                            lidar_node = next(iter(self.G_reduced.adj[node].items()))[0]

                            # Add all street points connected to the lidar to the deletion list
                            all_street_points = self.G_reduced.adj[lidar_node]
                            for street_point in all_street_points:
                                to_delete.append(street_point)
                            
                            # Add the lidar to the deletion list and radar1
                            to_delete.append(lidar_node)
                            self.lidar1.append(lidar_node)
        
        # Remove nodes marked for deletion
        for node_to_delete in list(set(to_delete)):
            self.G_reduced.remove_node(node_to_delete)

    def find_similar_lidar(self): 
        """
        Find lidars acting on the same street points and reduce the dimensionality of the Qubo Matrix
        """
        liders = []
        for lidar1 in self.G_reduced.nodes:
            if len(lidar1) != 3: #Checking that the node is indeed a lidar
                liders.append(lidar1)
        for lidar1 in liders:
            for lidar2 in liders:
                if lidar1 != lidar2:
                    if lidar1 in self.G_reduced.nodes and lidar2 in self.G_reduced.nodes:
                        adj1 = self.G_reduced.adj[lidar1]
                        adj2 = self.G_reduced.adj[lidar2]
                        #look if subset exists
                        if adj1.items() <= adj2.items():
                            self.reduce_q(lidar1)
                            self.lidar0.append(lidar1)
                        elif adj2.items() < adj1.items():
                            self.reduce_q(lidar2)
                            self.lidar0.append(lidar2)

    def reduce_q(self, lidar):
        self.G_reduced.remove_node(lidar)
        for edge in list(self.G_reduced.edges):
            if lidar in edge:
                self.G_reduced.remove_edge(edge[0], edge[1])

    def identify_isolated_nodes(self):
        """
        This function identifies isolated nodes in the graph and removes them.
        """
        isolated_nodes = list(nx.isolates(self.G))
        print(f"max_radiusIsolated nodes: {len(isolated_nodes)}")

    def get_num_variables(self):
        return len(self.listLidar)

    @classmethod
    def gen_problem(cls, num_cols, version, max_radius= 2.5, hor_basic_distance = 1, vert_basic_dist = 2):
        data_params = {
            "max_vert_angle": 30,
            "min_vert_angle": -80,
            "half_horizontal_angle": 180,
            "max_radius": max_radius,
            "lidar_wall_offset": 0.2
        }
        if version == 1:
            return cls._gen_problem(num_cols, [0], 0, (num_cols-1)*hor_basic_distance, 2.5, 0,-10, 0, (num_cols-1)*hor_basic_distance, vert_basic_dist, vert_basic_dist, 1, num_cols, **data_params)
        elif version == 2:
            return cls._gen_problem(num_cols, [0], 0, (num_cols-1)*hor_basic_distance, 2.5, 0, -10, 0, (num_cols-1)*hor_basic_distance, 0.5*vert_basic_dist, vert_basic_dist, 2, num_cols, **data_params)
        elif version == 3:
            return cls._gen_problem(num_cols, [0, 2*vert_basic_dist], 0, (num_cols-1)*hor_basic_distance, 2.5, 0, -10, 0, (num_cols-1)*hor_basic_distance, 0.5*vert_basic_dist, 1.5*vert_basic_dist, 3, num_cols, **data_params)
        else:
            print("Version can be ońly 1,2 or 3")

    @classmethod
    def create_problem_from_glb_file(cls, lidar_density, street_point_density):
        problem_dict = create_problem_from_glb(lidar_density=lidar_density, street_point_density=street_point_density)
        return cls.create_cls(problem_dict)


    @classmethod
    def from_json(cls, path):
        problem_dict = cls.__gimport(cls, path)
        return cls.create_cls(problem_dict)

    def __gimport(self, path):
        script_location = Path(__file__).resolve().parent
        relative_path = script_location / 'data'
        relative_path.mkdir(parents=True, exist_ok=True)
        path = relative_path / path
        with open(path) as surrounding:
            return json.load(surrounding)

    def create_graph_from_dict(self, problem_dict):
        self.problem_dict=problem_dict
        self.walls = self.__generateWalls()
        self.listLidar, self.listStreetPoints= self.__generateGraph()
        self.create_connections()
        self.preprocessing()

    @classmethod
    def create_cls(cls, problem_dict, data_params = {}):
        new_class = cls(**data_params)
        new_class.create_graph_from_dict(problem_dict)
        return new_class

    @classmethod
    def _gen_problem(cls, l_number_per_row, l_y, l_xmin, l_xmax, l_height, l_yaw_deg, l_pitch_deg, s_xmin, s_xmax, s_ymin, s_ymax, s_rows, s_cols, **data_params):
        problem_dict = cls.problem_generator(l_number_per_row, l_y, l_xmin, l_xmax, l_height, l_yaw_deg, l_pitch_deg, s_xmin, s_xmax, s_ymin, s_ymax, s_rows, s_cols)
        new_class = cls(**data_params)
        new_class.create_graph_from_dict(problem_dict)
        return new_class
    

    @classmethod
    def problem_generator(cls, l_number_per_row, l_y, l_xmin, l_xmax, l_height, l_yaw_deg, l_pitch_deg, s_xmin, s_xmax, s_ymin, s_ymax, s_rows, s_cols):

        lid=[]
        for i in l_y:
            lid.extend(cls.create_horizontal_lidar_points(l_number_per_row, i, l_xmin, l_xmax, l_height, l_yaw_deg, l_pitch_deg))
        
        str=cls.create_street_points(s_xmin, s_xmax, s_ymin, s_ymax, s_rows, s_cols)
        wal=[]
        problem_dict={'listLidar':lid, 'listCovering':str, 'wall':wal}

        return problem_dict
    

    @classmethod
    def create_horizontal_lidar_points(cls, number, y, xmin, xmax, height, yaw_deg, pitch_deg): 
        res=[]
        x=[]
        x.extend(np.linspace(xmin, xmax, number))
        ytemp=[]
        ytemp.extend(np.linspace(xmin, xmax, number)*0+ y)
        pos=list(zip(x,ytemp))
        for l in pos: 
            res.append((l[0], l[1], height, yaw_deg, pitch_deg))
        return res
    
    @classmethod
    def create_street_points(cls, xmin, xmax, ymin, ymax, rows, cols): 
            res=[]
            yloop=np.linspace(ymin,ymax,rows)
            x=[]
            for item in yloop:
                y=np.linspace(xmin, xmax, cols)*0+ item
                res.extend(y)
                x.extend(np.linspace(xmin, xmax, cols))
            return list(zip(x,res))

    def create_connections(self): 
        edgecount=0
        for s in self.listStreetPoints:
            nc=1
            for l in self.listLidar:
                line=(l,s)
                if (self._in_range(l,s, self.max_radius, self.max_vert_angle, self.min_vert_angle, self.half_horizontal_angle)):
                    inters=0
                    for w in self.walls:
                        inters+=self._intersect(line,w) 
                        if inters: 
                            break   
                    if inters==0: 
                        self.G.add_edge((s[0], s[1], s[2]),(l[0], l[1], l[2], l[3], l[4]))
                        edgecount+=1
                        nc=0
            if nc:
                self.listStreetPointsNeverCovered.append(s) 
        self.never_covered=len(self.listStreetPointsNeverCovered) 
             
    def __generateGraph(self):
        pointsL3D = []
        pointsS3D = []
        for i in self.problem_dict['listLidar']:
            pointsL3D.append((i[0], i[1], i[2], i[3], i[4]))
        
        for w in self.problem_dict['wall']:
 
            if w[3]>0:
                pitch=w[7]
                #Verbindungsvektor Maueranfang zu Ende
                diff=[w[1][0]-w[0][0], w[1][1]-w[0][1]]
                #Länge der Mauer
                length=math.sqrt(diff[0]**2+diff[1]**2)
                #Abstandsvektor senkrecht zur Mauer
                perpendicular_offset=[diff[1]*w[5]/length,-diff[0]*w[5]/length]
                #Verbindungsvektor Lidarstrecke Anfang zu Ende, ist um 2*lidar_wall_offset kürzer als diff
                difflidar=[diff[0]*(1-2*self.lidar_wall_offset/length),diff[1]*(1-2*self.lidar_wall_offset/length)]
                
                difflidarlength=math.sqrt(difflidar[0]**2+difflidar[1]**2)
                #Anzahl der zu plazierenden Lidare
                #Reale Lidardichte soll höchstens die vorgegebene sein
                numlid=int(np.floor(difflidarlength*w[3]))
                #Mindestens 1 Lidar setzen
                if numlid <=1: 
                    numlid=1
            

                difflidarstart=[w[0][0]+self.lidar_wall_offset/length*diff[0]+perpendicular_offset[0], w[0][1]+self.lidar_wall_offset/length*diff[1]+perpendicular_offset[1]]
                if numlid>1:
                    for i in range(0, numlid):   
                        #numlid lidare werden gesetzt                    
                        lx=difflidarstart[0]+i/(numlid-1)*difflidar[0]
                        ly=difflidarstart[1]+i/(numlid-1)*difflidar[1]
                        lz=w[4]
                        ang=90-math.atan2(perpendicular_offset[1], perpendicular_offset[0])*180/math.pi
                        if w[6]>0: 
                            #Alternating mode
                            ang+=((i+w[6])%2-0.5)*2*(90-self.half_horizontal_angle)
                        #Neues Mauerradar hinzufügen
                        pointsL3D.append((lx,ly,lz, ang, pitch))
                else: 
                    #numlid=1
                    # ein Lidar in der Mitte
                    lx=w[0][0]+diff[0]*0.5+perpendicular_offset[0]
                    ly=w[0][1]+diff[1]*0.5+perpendicular_offset[1]
                    lz=w[4]
                    ang=90-math.atan2(perpendicular_offset[1], perpendicular_offset[0])*180/math.pi
                    if w[6]>0: 
                        #Alternating mode
                        ang+=((i+w[6])%2-0.5)*2*(90-self.half_horizontal_angle)
                    #Neues Mauerradar hinzufügen
                    pointsL3D.append((lx,ly,lz, ang, pitch))
                
        for i in self.problem_dict['listCovering']:
            pointsS3D.append((i[0], i[1], 0))
            
        self.G.add_nodes_from(pointsL3D)
        self.G.add_nodes_from(pointsS3D)

        return pointsL3D, pointsS3D

    def __generateWalls(self):
        walls=[]
        walls3D=[]
    
        for i in self.problem_dict['wall']:
            walls.append((i[0][0], i[0][1]))
            walls.append((i[1][0], i[1][1]))
            walls3D.append((i[0], i[1], i[2], i[3]))
            self.M.add_edge((i[0][0], i[0][1]), (i[1][0], i[1][1]))
        self.M.add_nodes_from(walls)
        return walls3D
    
    @classmethod
    def _in_range(cls, l,s, maxcoverage_meter, max_vert_angle, min_vert_angle, half_horizontal_angle):
        too_far=math.sqrt((s[0]-l[0])**2+   (s[1]-l[1])**2)>maxcoverage_meter
        if too_far: 
            return 0
        
        #relative orientation of street point in the system of the lidar with yaw and pitch
        
        #angle convention north/clockwise transformed to mathematical angle convention
        ya=(90-l[3])/180*math.pi
        #rotation matrix for yaw
        #if yaw positive (counterclockwise), relative movement of streetpoint is clockwise
        y=np.array([[math.cos(ya), math.sin(ya), 0], 
           [-math.sin(ya), math.cos(ya), 0],
           [0,0,1]
           ])
        
        pitch_angle=l[4]/180*math.pi
        #if pitch of lidar upwards, streetpoint is moving downwards
        p=np.array([[math.cos(pitch_angle), 0, math.sin(pitch_angle)], 
           [0, 1, 0],
           [-math.sin(pitch_angle), 0, math.cos(pitch_angle)]
           ])
        total_rotation=np.matmul(p,y)
        la=np.array(l[0:3])
        sa=np.array(s)
        
      
        rel_rotated=np.matmul(total_rotation, sa-la)
        
        #decision if streetpoint in vertical FoV
        vert_angle=math.atan2(rel_rotated[2], math.sqrt(rel_rotated[0]**2+rel_rotated[1]**2))
        if vert_angle>max_vert_angle/180*math.pi or vert_angle<min_vert_angle/180*math.pi: 
            return 0
        
        #decision if streetpoint in horizontal FoV
        hor_angle=math.atan2(rel_rotated[1], rel_rotated[0])
        if math.fabs(hor_angle)>half_horizontal_angle/180*math.pi: 
            return 0
       
        return 1

    @classmethod
    def _intersect(cls, line, w):
        
        a=line[1][0]-line[0][0]
        b=w[0][0]-w[1][0]
        c=line[1][1]-line[0][1]
        d=w[0][1]-w[1][1]
        det=a*d-b*c
        if det==0: 
            return 0
        line_s=line[0]
        lidarh=line_s[2]
        mauerh=w[2]
        if mauerh==0: 
            return 0
        ws=w[0] #eckige Klammer ist Liste ist Vektor
        diff=[ws[i]-line_s[i] for i in range(len(ws))]
        im=1.0/det*np.array([[d,-b],[-c,a]])
        r=np.dot(im,diff)
        if not (r[0]>0 and r[0]<1 and r[1]>0 and r[1]<1): 
            return 0

        #r[0] Anteil zwischen Lidar und Mauer vergl. zu Lidar und Straßenpunkt, wenn in line erst das lidar kommt
        if lidarh/mauerh>=1/(1-r[0]): 
            return 0
        else: 
            return 1

    @classmethod
    def from_random(cls, *args, **kwargs):
        raise NotImplementedError



