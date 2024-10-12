import glob
import numpy as np

###############################################################################
###############################################################################
# Load all graph of the letters
###############################################################################
###############################################################################
path: str = '../PrePro/Complete_Graph/Graph_segments/Letters/110b/*.csv'
graph_names = glob.glob(path, recursive=True)

for graph in graph_names:
    edges = np.loadtxt(graph, delimiter=",", dtype=int, skiprows=1, usecols=(1, 2))

###############################################################################
###############################################################################
# Load the original BBs
###############################################################################
###############################################################################
path_main = str('../PrePro/Complete_Graph/')
path2 = path_main + str('*.jpg')
image_names = glob.glob(path2, recursive=True)

# Function to read the text file and create an array
def replace_n(line):
  return line.replace('\n','').split(' ')

# Function to convert the bounding boxes (bb) coordinates to pixels coord.
def conv_bb(line):
  # Here it is step + the BB 
  x_1 = int(float(line[1]))*640 + ((int((float(line[1])-int(float(line[1]))) * 640)) - int(((float(line[3])-int(float(line[3]))) * 640)/2))
  y_1 = int(float(line[2]))*640 + ((int((float(line[2])-int(float(line[2]))) * 640)) - int(((float(line[4])-int(float(line[4]))) * 640)/2))
  x_2 = int(float(line[3]))*640 + ((int((float(line[1])-int(float(line[1]))) * 640)) + int(((float(line[3])-int(float(line[3]))) * 640)/2))
  y_2 = int(float(line[4]))*640 + ((int((float(line[2])-int(float(line[2]))) * 640)) + int(((float(line[4])-int(float(line[4]))) * 640)/2))
  coord_bb = [[x_1, y_1], [x_2, y_2]]
  return coord_bb

# GLoad all label names for the txt files
label_names: list = []
for image_name in image_names:
    label = image_name.replace('images','labels', 1).replace('.jpg','.txt', 1)
    label_names.append(label)

###############################################################################
###############################################################################
# Create a vector of BBs and IDs
BBs: list = []
IDs: list = []
for idx, label in enumerate(label_names): 
    ID_obj: int = 0
    with open(label,'r') as f:      # Load all images
        label = f.readlines()
    '''
    # For the BB
    for line in label:
        line = replace_n(line)  # Change str information to vector
        c_bb = conv_bb(line)    # Convert the coordinates of the BB
        BBs.append(c_bb) 
    '''
    # for the IDs
    for line in label:
        line = replace_n(line)  # Change str information to vector
        # Letters [0 to 9, a to z, A to Z] 	[0 to 61]
        # From a to z [0 - 9]
        for i in range(0,10):
            if line[0] == str(i):
                IDs.append([ID_obj, str(i)])
        # From a to z [10 - 35]
        for i in range(0,26):
            if line[0] == str(10+i):
                IDs.append([ID_obj, chr(97+i)])
        # From A to Z [36 - 61]
        for i in range(0,26):
            if line[0] == str(36+i):
                IDs.append([ID_obj, chr(65+i)])
        ID_obj +=1
#np.sort(IDs, axis=0, order=None)  
IDs = np.array(IDs)   
###############################################################################
############################################################################### 

edges=[(0, 1),
 (1, 2),
 (2, 3),
 (5, 6),
 (6, 7),
 (30, 31),
 (31, 32),
 (35, 36),
];

p1=[]
for i in range(0,len(edges)):
    for j in range(0,len(IDs)):
        if str(edges[i][0]) == IDs[j][0]:
            print(edges[i][0], IDs[j][0], IDs[j])
            
            p1.append(IDs[j][1])

p2=[]
for i in range(0,len(edges)):
    for j in range(0,len(IDs)):
        if str(edges[i][1]) == IDs[j][0]:
            print(edges[i][1], IDs[j][0], IDs[j])
            
            p2.append(IDs[j][1])   
            #p = np.char.replace(str(edges[i]), str(edges[i][0]), IDs[j][1])
x=(p1,p2)
x = np.reshape(x, (7, 2))



import networkx as nx
def print_graph_info(G):
  print("Directed graph:", G.is_directed())
  print("Number of nodes:", G.number_of_nodes())
  print("Number of edges:", G.number_of_edges())

H = []
H = nx.DiGraph()

for i in range(0, len(edges)):
  H.add_nodes_from([(int(edges[i][0]), {"color": "red", "size": 100}),])
for i in range(0, len(edges)):
  H.add_nodes_from([(int(edges[i][1]), {"color": "red", "size": 100}),])

node_colors = nx.get_node_attributes(H, "color").values()
colors = list(node_colors)
node_sizes = nx.get_node_attributes(H, "size").values()
sizes = list(node_sizes)

H.add_edges_from(edges)
G = H.to_undirected()
print_graph_info(G)
#print(G.edges())

nx.draw(G, with_labels=True, node_color=colors, node_size=sizes)

import networkx
from networkx import (
    draw,
    DiGraph,
    Graph,
)

b=[]
for component in networkx.connected_components(G):
    b.append(component)
    print(component)
