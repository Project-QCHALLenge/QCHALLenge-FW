from pygltflib import GLTF2
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime
from pathlib import Path

from scipy.spatial.transform import Rotation as R

def transform_coordinates_gltf_to_visualization(coords):
    """Transforms coordinates from glTF (Y up) to visualization system (Z up)."""
    if len(coords) == 3:
        x, y, z = coords  # glTF: X right, Y up, Z out of screen
        return [-x, z, y]
    if len(coords) == 4:
        w, x, y, z = coords
        return [-x,z,y,w]

def create_object(data, mesh, xmintotal, xmaxtotal, ymintotal, ymaxtotal, walldir):
    """Creates a vertical wall and adds it to the 3D plot using mesh dimensions."""
    mesh_id = mesh["mesh"]
    accessor_id = data["meshes"][mesh_id]["primitives"][0]["attributes"]["POSITION"]
    accessor = data['accessors'][accessor_id]
    
    # Extract min and max values to determine size
    min_vals = transform_coordinates_gltf_to_visualization(accessor['min'])
    max_vals = transform_coordinates_gltf_to_visualization(accessor['max'])
    # Max and Min Vals [0] are swapped due to the rotated coord system
    maxtemp=min_vals[0]
    min_vals[0]=max_vals[0]
    max_vals[0]=maxtemp


    # Calculate the size of the mesh
    size_x = max_vals[0] - min_vals[0]
    size_y = max_vals[1] - min_vals[1]
    size_z = max_vals[2] - min_vals[2]

    center_x = (min_vals[0] + max_vals[0]) / 2
    center_y = (min_vals[1] + max_vals[1]) / 2
    center_z = (min_vals[2] + max_vals[2]) / 2
    center = [center_x, center_y, center_z]

    # Adjust for the center position based on translation
    translation = mesh.get("translation", [0, 0, 0])
    x, y, z = transform_coordinates_gltf_to_visualization(translation)

    center = np.array(center) + np.array([x,y,z])

    rotation = mesh.get("rotation", [0, 0, 0, 1])  # Default to no rotation
    rotation = transform_coordinates_gltf_to_visualization(rotation)
    
    rot = R.from_quat(rotation)
    
    # Define vertices of the wall polygon based on the actual mesh size
    if mesh["name"] == "WallMesh":
        vertices = np.array([
            walldir%2*(x + size_x/2, y - size_y/2, z + size_z)+(1-walldir%2)*(x - size_x/2, y + size_y/2, z + size_z), 
            (1-walldir%2)*(x + size_x/2, y - size_y/2, z + size_z)+walldir%2*(x - size_x/2, y + size_y/2, z + size_z)
        ])
        
    else:
        vertices = np.array([
            [x - size_x/2, y - size_y/2, z],
            [x - size_x/2, y - size_y/2, z + size_z],
            [x + size_x/2, y - size_y/2, z + size_z],
            [x + size_x/2, y - size_y/2, z],
            [x - size_x/2, y + size_y/2, z],
            [x - size_x/2, y + size_y/2, z + size_z],
            [x + size_x/2, y + size_y/2, z + size_z],
            [x + size_x/2, y + size_y/2, z]
        ])
        
    translated_vertices = [vertex - center for vertex in vertices]
    rotated_vertices = rot.apply(translated_vertices)
    vertices = [vertex + center for vertex in rotated_vertices]


    # Create faces of the polygon
    """  verts = [[vertices[0], vertices[1], vertices[5], vertices[4]], 
             [vertices[7], vertices[6], vertices[2], vertices[3]], 
             [vertices[0], vertices[3], vertices[2], vertices[1]],
             [vertices[4], vertices[5], vertices[6], vertices[7]],
             [vertices[1], vertices[2], vertices[6], vertices[5]],
             [vertices[4], vertices[7], vertices[3], vertices[0]]] 
             """

    # To check, if always correct?
    basevertslist=[]

    if mesh["name"] == "WallMesh":
        baseverts = [vertices[0], vertices[1]]
        basevertslist.append([[baseverts[0][0], baseverts[0][1]],[baseverts[1][0], baseverts[1][1]]])
    else: 
        baseverts = [vertices[1], vertices[2], vertices[6], vertices[5]]
        basevertslist.append([[baseverts[0][0], baseverts[0][1]],[baseverts[1][0], baseverts[1][1]]])
        basevertslist.append([[baseverts[1][0], baseverts[1][1]],[baseverts[2][0], baseverts[2][1]]])
        basevertslist.append([[baseverts[2][0], baseverts[2][1]],[baseverts[3][0], baseverts[3][1]]])
        basevertslist.append([[baseverts[3][0], baseverts[3][1]],[baseverts[0][0], baseverts[0][1]]])
    
    
    xmax=-10000
    xmin=10000
    ymax=-10000
    ymin=10000

    # Add the polygon to the plot
    #ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='green', alpha=.25))
    #ax.add_collection3d(Poly3DCollection([baseverts], linewidths=1, edgecolors='red', alpha=.25))
    for b in basevertslist: 
        for m in b: 
            xmax=max(xmax,np.array(m).T[0].max())
            xmin=min(xmin,np.array(m).T[0].min())
            ymax=max(ymax,np.array(m).T[1].max())
            ymin=min(ymin,np.array(m).T[1].min())
    
    return min(xmin, xmintotal), max(xmax, xmaxtotal), min(ymin, ymintotal), max(ymax, ymaxtotal), basevertslist, size_z

def create_street_points(xmin, xmax, ymin, ymax, xmargin, ymargin, density): 
    res=[]
    yloop=np.linspace(ymin+ymargin,ymax-ymargin,math.ceil((ymax-ymin)*density))
    x=[]
    for item in yloop:
        y=np.linspace(xmin+xmargin, xmax-xmargin, math.ceil((xmax-xmin)*density))*0+ item
        res.extend(y)
        x.extend(np.linspace(xmin+xmargin, xmax-xmargin, math.ceil((xmax-xmin)*density)))
    
    return list(zip(x,res))

def create_problem_from_glb(lidar_density = 0.1, street_point_density = 0.1, save_as_json = False, show_plot = False):

    lidar_height=2.5
    lidar_lateral_offset=0.2
    lidar_direction_mode=0
    lidar_pitch=-30
    pfostenlimit=2
    #Strassenpunkte
    xmargin=0.5
    ymargin=0.5

    script_dir = Path(__file__).parent

    # Construct the full path to the GLB file
    filename = script_dir / "data" / "simObjectsExport_utc_2023_5_11.glb"

    # Load the GLB file
    glb = GLTF2().load(str(filename))

    json_string = glb.to_json()
    data = json.loads(json_string)

    walls = []
    for mesh in data["nodes"]:
        if mesh["name"] == "WallMesh":
            walls.append(mesh)

    boxes = []
    for mesh in data["nodes"]:
        if mesh["name"] == "Box":
            boxes.append(mesh)

    xmintotal=10000
    xmaxtotal=-10000
    ymintotal=10000
    ymaxtotal=-10000

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    wallcount=0
    lidarwalls=[]
    for wall in walls:
        wallcount+=1
        xmintotal, xmaxtotal, ymintotal, ymaxtotal, basevertslist, size_z=create_object(data, wall, xmintotal, xmaxtotal, ymintotal, ymaxtotal, wallcount)
        for wal in basevertslist: 
            wal.extend([size_z, lidar_density, lidar_height, lidar_lateral_offset, lidar_direction_mode, lidar_pitch])
            lidarwalls.append(wal)

    for box in boxes:
        xmintotal, xmaxtotal, ymintotal, ymaxtotal, basevertslist, size_z=create_object(data, box, xmintotal, xmaxtotal, ymintotal, ymaxtotal, 0)
        for wal in basevertslist: 
                wal.extend([size_z, (size_z > pfostenlimit)* lidar_density, lidar_height, lidar_lateral_offset*(2*(size_z <= pfostenlimit)-1), lidar_direction_mode, lidar_pitch])
                lidarwalls.append(wal)

    # if show_plot:
    #     ax.set_xlim([-60, 60])
    #     ax.set_ylim([-60, 60])
    #     ax.set_zlim([0, 120])
    #     plt.show()
    # plt.clf()

    sp=create_street_points(xmintotal, xmaxtotal, ymintotal, ymaxtotal, xmargin, ymargin, street_point_density)
    lid= []

    problem_dict = {'listCovering':sp, 'wall':lidarwalls, 'listLidar': lid}

    if save_as_json:
        now = datetime.now()    
        date_time = now.strftime("_%Y_%m_%d_%H_%M_%S")   
        outfile="data/data/tmp"+(filename.replace('/','_')).replace('.','_')+"_lid"+str(lidar_density).replace('.','p')+'_sp'+str(street_point_density).replace('.','p')+date_time+'.json'
        with open(outfile, "w") as file_write:
            json.dump(problem_dict, file_write)

    #print('GLB_reader finished!')

    return problem_dict
