# Importing required libraries
import numpy as np
import networkx as nx
import networkx.drawing.nx_pylab as nxplot
import random



# Initial points
init = [(11, 12), (12, 1), (19, 6)]    

            
def connect_neigh(point, lake_map, G):
    if lake_map[point[0] - 1, point[1] - 1] != 0:
        G.add_edge(lake_map[point[0], point[1]], lake_map[point[0] - 1, point[1] - 1], weight = random.random())
    if lake_map[point[0] - 1, point[1]] != 0:
        G.add_edge(lake_map[point[0], point[1]], lake_map[point[0] - 1, point[1]], weight = random.random())
    if lake_map[point[0] - 1, point[1] + 1] != 0:
        G.add_edge(lake_map[point[0], point[1]], lake_map[point[0] - 1, point[1] + 1], weight = random.random())
    if lake_map[point[0] + 1, point[1] - 1] != 0:
        G.add_edge(lake_map[point[0], point[1]], lake_map[point[0] + 1, point[1] - 1], weight = random.random())
    if lake_map[point[0] + 1, point[1]] != 0:
        G.add_edge(lake_map[point[0], point[1]], lake_map[point[0] + 1, point[1]], weight = random.random())
    if lake_map[point[0] + 1, point[1] + 1] != 0:
        G.add_edge(lake_map[point[0], point[1]], lake_map[point[0] + 1, point[1] + 1], weight = random.random())
    if lake_map[point[0], point[1] - 1] != 0:
        G.add_edge(lake_map[point[0], point[1]], lake_map[point[0], point[1] - 1], weight = random.random())
    if lake_map[point[0], point[1] + 1] != 0:
        G.add_edge(lake_map[point[0], point[1]], lake_map[point[0], point[1] + 1], weight = random.random())
    return G
    
def create_graph(lake_map, importance_map):
    # Generating graph
    G = nx.DiGraph()
    # Adding edges
    for x_index in range(lake_map.shape[0]):
        for y_index in range(lake_map.shape[1]):
            if lake_map[x_index, y_index] != 0:
                G = connect_neigh([x_index, y_index], lake_map, G)
    # Adding xy coordinates    
    for x_index in range(lake_map.shape[0]):
        for y_index in range(lake_map.shape[1]):
            if lake_map[x_index, y_index] != 0:
                G.nodes[lake_map[x_index, y_index]]['xy'] = [x_index, y_index]
    # Adding importance    
    for x_index in range(lake_map.shape[0]):
        for y_index in range(lake_map.shape[1]):
            if lake_map[x_index, y_index] != 0:
                G.nodes[lake_map[x_index, y_index]]['importance'] = [item[x_index, y_index] for item in importance_map]
    # Adding idleness
    for x_index in range(lake_map.shape[0]):
        for y_index in range(lake_map.shape[1]):
            if lake_map[x_index, y_index] != 0:
                G.nodes[lake_map[x_index, y_index]]['idle'] = 0
    return G

def plot_graph(G, lake_map, sol = None):
    max_val = int(np.max(lake_map))
    pos = {}
    for index in range(1, max_val + 1):
        a = np.where(lake_map == index)
        pos[index] = np.array([int(a[0][0]), int(a[1][0])])   
    nx.draw(G, with_labels='true', pos = pos)
    if sol:
        nx.draw_networkx_nodes(G, pos = pos, nodelist = sol, node_color = 'r')

def change_weights(G):
    info = list(G.edges(data=True))
    for item in info:
        G[item[0]][item[1]]['weight'] = random.random()
    return G
    
def create_path(Gin, lake_map, initial_point, n_points):
    G = change_weights(Gin)
    max_val = int(np.max(lake_map))
    path = list()
    current_point = initial_point
    for p in range(n_points - 1):
        final_point = random.randint(1, max_val)
        path.extend(nx.dijkstra_path(G, current_point, final_point)[:-1]) 
        current_point = final_point
    final_point = initial_point
    path.extend(nx.dijkstra_path(G, current_point, final_point)) 
    return list(map(int, path))

def crossover_2p(G, path1, path2):
    point1a = random.randint(1, len(path1)-1 )
    point1b = random.randint(1, len(path1)-1 )
    point2a = random.randint(1, len(path2)-1 )
    point2b = random.randint(1, len(path2)-1 )
    if point1b < point1a:
        aux = point1b 
        point1b = point1a
        point1a = aux   
    if point2b < point2a:
        aux = point2b 
        point2b = point2a
        point2a = aux
    bridge1 = list(map(int, nx.dijkstra_path(G, path1[point1a], path2[point2b]) ))
    bridge2 = list(map(int, nx.dijkstra_path(G, path2[point2a], path1[point1b]) ))
    path1b  = path1[point1b:]
    path2b  = path2[point2b:]
    del path1[-len(path1[point1a:]):]
    del path2[-len(path2[point2a:]):]
    path1.extend(bridge1)
    path2.extend(bridge2)
    path1.extend(path2b[1:])
    path2.extend(path1b[1:])
    return path1, path2

def crossover_2p_nship(G, path1, path2, n_ship):
    # Number of movements of each ship for each individual
    pointer1 = 1
    pointer2 = 1
    # for each couple of ships
    for index in range(n_ship):
        path1cross, path2cross = crossover_2p(G, path1[pointer1 : pointer1 + path1[pointer1-1]], path2[pointer2 : pointer2 + path2[pointer2-1]])
        path1[pointer1 : pointer1 + path1[pointer1-1]] = []
        path2[pointer2 : pointer2 + path2[pointer2-1]] = []
        path1[pointer1 : pointer1] = path1cross
        path2[pointer2 : pointer2] = path2cross
        path1[pointer1-1] = len(path1cross)
        path2[pointer2-1] = len(path2cross)
        pointer1 = pointer1 + len(path1cross)+1
        pointer2 = pointer2 + len(path2cross)+1
    return path1, path2
    
def mutate(G, path):
    intersection = list()
    while len(intersection) == 0:
        mut_index = random.randint(1, len(path)-2)
        node     = path[mut_index]
        node_pre = path[mut_index - 1]
        node_pos = path[mut_index + 1]
        intersection = list(set(G.neighbors(node_pre)).intersection(G.neighbors(node_pos)))
        intersection.remove(node)
    path[mut_index] = int(random.choice(intersection))
    return path, mut_index
   
def mutate_nship(G, path, n_ship, indpb):
    # Remove non-mutable genes
    num_range = list(range(len(path)))
    pointer = 0
    for ship in range(n_ship):
        size = path[pointer]
        num_range.remove(pointer)
        num_range.remove(pointer+1)
        pointer += size
        num_range.remove(pointer)
        pointer += 1
    # Mutate
    for mut_index in num_range:
        if random.random() < indpb:
            node     = path[mut_index]
            node_pre = path[mut_index - 1]
            node_pos = path[mut_index + 1]
            intersection = list(set(G.neighbors(node_pre)).intersection(G.neighbors(node_pos)))            
            intersection.remove(node)
            if len(intersection) != 0:                
                path[mut_index] = int(random.choice(intersection))
    return path, mut_index


    
