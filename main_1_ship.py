# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
from time import time
from ASV import fleet
from graph_functions import init, create_graph, create_path, mutate, mutate_nship, crossover_2p, crossover_2p_nship, plot_graph

# Importing lake topology and importance map
importance_map = [ np.genfromtxt('Data/shekel_gt.csv', delimiter=',') ]
lake_map = np.genfromtxt('Data/map.csv', delimiter=',')
val = 1
for x_index in range(lake_map.shape[0]):
    for y_index in range(lake_map.shape[1]):
        if lake_map[x_index, y_index] == 1:
            lake_map[x_index, y_index] = val
            val += 1
G = create_graph(lake_map, importance_map)

# Instance of fleet
n_ships = 1
fleet1 = fleet( nship = n_ships,
                lake_map = lake_map,
                graph = G,
                initial_point = [lake_map[item] for item in init] )

# evaluate individual
def evalTrajectory(ind):
    fleet1.reset()
    fleet1.set_trajectory(ind)
    return fleet1.move()


# Plotting convergence of the algorithm
def plot_evolution(log):

    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")

    plt.plot(gen, fit_mins, "b")
    plt.plot(gen, fit_maxs, "r")
    plt.plot(gen, fit_ave, "--k")
    plt.fill_between(gen, fit_mins, fit_maxs, facecolor = "g", alpha = 0.2)
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.legend(["Min", "Max", "Avg"])
    plt.grid()
    plt.savefig("Results/msso.pdf", dpi = 300)
    
# Creation of the algorithm: maximization
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', list, fitness = creator.FitnessMax)

# Genetic operations
def create_individual():
    return create_path(G, lake_map, lake_map[init[0]], 4)

def create_individual_n_ships():
    paths = list()
    for index in range(n_ships):
        auxiliar = create_path(G, lake_map, lake_map[init[index]], 4)
        paths.append(len(auxiliar))
        paths += auxiliar
    return paths

#% [len(item) for item in paths[:-1]] + [y for x in paths for y in x]

def crossover(path1, path2):
    return crossover_2p(G, path1, path2)

def crossover_n_ships(path1, path2):
    return crossover_2p_nship(G, path1, path2, n_ships)
    
def mutate_ind(path, indpb):
    if random.random() < indpb:
        return mutate(G, path)[0],
    else:
        return path,
    
def mutate_ind_n_ships(path, indpb):
    return mutate_nship(G, path, n_ships, indpb)[0],

# Creation of the toolbox 
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual_n_ships)
toolbox.register("ini_population", tools.initRepeat, list, toolbox.individual) 
toolbox.register("mate", crossover_n_ships)
toolbox.register("mutate", mutate_ind_n_ships, indpb = 0.05)
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("evaluate", evalTrajectory)


# single_objective function
def single_objective(cxpb, mutpb, ngen, i):
    random.seed(i)
    CXPB, MUTPB, NGEN = cxpb, mutpb, ngen
    pop = toolbox.ini_population(500)
    MU, LAMBDA = len(pop), len(pop)
    hof = tools.HallOfFame(1, similar = np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    t0 = time()
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU,
                                             LAMBDA, CXPB, MUTPB,
                                             NGEN, stats = stats,
                                             halloffame = hof,
                                             verbose = True)
    print("Time: {} s".format(time() - t0))
    return pop, hof, logbook

#%%
# pop, hof, logbook = single_objective(0.5, 0.5, 1500, 1)
# print(hof[0])
# plot_evolution(logbook)


#%%
parameters= [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
ngen = 1000
for cxpb, mutpb in parameters:
    CXPB, MUTPB, NGEN = cxpb, mutpb, ngen
    for i in range(0, 30):
        res_individuos = open('Results/ind_single_' + str(n_ships) + 'ships.txt', 'a')
        res_fitness = open('Results/fitness_single_' + str(n_ships) + 'ships.txt', 'a')
        res_fitness.write("id,cx,mu,ft")
        res_fitness.write("\n")
        pop_new, pareto_new, log = single_objective(cxpb, mutpb, ngen, int(i))
        for ide, ind in enumerate(pareto_new):
            res_individuos.write(str(i))
            res_individuos.write(",")
            res_individuos.write(str(ind))
            res_individuos.write("\n")
            res_fitness.write(str(i))
            res_fitness.write(",")
            res_fitness.write(str(cxpb))
            res_fitness.write(",")
            res_fitness.write(str(mutpb))
            res_fitness.write(",")
            res_fitness.write(str(ind.fitness.values[0]))
            print(ind.fitness.values[0])
            res_fitness.write("\n")
        del(pop_new)
        del(pareto_new)
        res_fitness.close()
        res_individuos.close()  




