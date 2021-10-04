import numpy as np
import random

class fleet:
    def __init__(self, nship, lake_map, graph, initial_point):
        # Graph with all the information (including idle and importance)
        self.base_graph = graph
        self.lake_map = lake_map
        self.graph = graph
        
        # Generating fleet
        self.ships = list()
        self.nships = 0
        for index in range(nship):
            self.ships.append(ASV(lake_map, initial_point[index]))
            self.nships += 1
            
    def reset(self):
        for ship in self.ships:
            ship.reset()  
        self.graph = self.base_graph.copy()
    
    def set_trajectory(self, ind):
        # Setting trajectories to the fleet
        pointer = 1
        for index, ship in enumerate(self.ships):
            ship.set_trajectory(ind[pointer:pointer + ind[pointer-1]])
            pointer += ind[pointer-1] + 1
                
    def move(self):       
        # Maximum number of steps and rewards initialization
        max_steps = np.max([len(ship.trajectory) for ship in self.ships])
        reward = np.array([0]*len(self.graph.nodes[1]['importance']), dtype = float)
        
        # For each step
        for step in range(1, max_steps):
                        
            # Updating the position of the ships
            pos = list()
            for ship in self.ships:
                if ship.pointer < len(ship.trajectory) - 1:                    
                    pos.append(ship.step())
                    
            # Evaluate collisions
            if (len(pos) == len(set(pos))) == False:
                reward -= np.array([100]*len(self.graph.nodes[1]['importance']), dtype = float)
           
            # Computing rewards
            idle = [self.graph.nodes[node]['idle'] for node in pos]
            imp = [self.graph.nodes[node]['importance'] for node in pos]
            for imp_index in range(len(imp[0])):
                for ship_index in range(len(idle)):
                    reward[imp_index] += np.array(idle[ship_index])*np.array(imp[ship_index][imp_index])                                       
                     
            # Updating idle and importance values
            for node in range(1, len(self.graph)):
                if node in pos:
                    self.graph.nodes[node]['idle'] = 0
                    self.graph.nodes[node]['importance'] = list(np.array(self.graph.nodes[node]['importance']) - 0.2*np.array(self.base_graph.nodes[node]['importance']))
                    for index in range(len(self.graph.nodes[node]['importance'])):
                        if self.graph.nodes[node]['importance'][index] < 0:
                            self.graph.nodes[node]['importance'][index] = 0
                else:
                    self.graph.nodes[node]['idle'] = self.graph.nodes[node]['idle'] + 1               
            
        # Checking total distance
        for ship in self.ships:
            if ship.n_movements > 39.5:
                return tuple([-1]*len(reward))
        
        # Retruning the total reward
        return tuple(reward)
        
        
class ASV:
    def __init__(self, lake_map, initial_point):
        self.lake_map = lake_map
        # Initial point and actual point
        self.initial_point = initial_point
        self.actual_point = initial_point
        # Movements 
        self.n_movements = 0
        # Trajectory
        self.trajectory = None
        # Pointer to position
        self.pointer = 0
        
    def reset(self):
        # Initial point and actual point
        self.actual_point = self.initial_point
        # Movements 
        self.n_movements = 0
        # Trajectory
        self.trajectory = None
        # Pointer
        self.pointer = 0
        
    def set_trajectory(self, ind):
        self.trajectory = ind
        
    def step(self):
        # Updating pointer and position
        self.pointer += 1
        self.actual_point = self.trajectory[self.pointer]
        # Upadting distance
        aux = np.where(self.lake_map == self.trajectory[self.pointer - 1])
        xy_pre = [aux[0][0], aux[1][0]]
        aux = np.where(self.lake_map == self.trajectory[self.pointer])
        xy_post = [aux[0][0], aux[1][0]]
        if np.sum(np.abs(np.array(xy_pre) - np.array(xy_post))) == 2:
            self.n_movements += 1
        else:
            self.n_movements += 1/np.sqrt(2)
        return self.actual_point
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        