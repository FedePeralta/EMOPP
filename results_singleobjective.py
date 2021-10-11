import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

'''
    Indicate the number of ships using:
        number_of_ships = 1
    Works with 1, 2 and 3 ships.
    
    Run the script to observe the best individual obtained by the system. 
'''
number_of_ships = 3


def get_clean(_file):
    for nnan in nans:
        _file[nnan[1], nnan[0]] = np.nan
    return np.ma.array(_file, mask=np.isnan(_file))


f3 = pd.read_csv(f"Results/fitness_single_{number_of_ships}ships.txt")
ind = []
with open(f"Results/ind_single_{number_of_ships}ships.txt") as f:
    line = f.readline()
    while line != "":
        full_ind = np.array([int(i) for i in line[line.find(",") + 2:-2].split(',')])
        ind.append(full_ind)
        line = f.readline()

m = pd.read_csv("Data/map.csv", names=np.arange(0, 19, 1)).to_numpy()
m_vect2 = np.flipud(np.asarray(np.where(m == 1)).reshape(2, -1)).T
nans = np.flipud(np.asarray(np.where(m == 0)).reshape(2, -1)).T
m1 = get_clean(pd.read_csv("Data/shekel_gt.csv", names=np.arange(0, 19, 1)).to_numpy())
m2 = get_clean(pd.read_csv("Data/rosenbrock_gt.csv", names=np.arange(0, 19, 1)).to_numpy())
m3 = get_clean(pd.read_csv("Data/himmelblau_gt.csv", names=np.arange(0, 19, 1)).to_numpy())

X, Y = np.meshgrid(np.arange(0, 19, 1), np.arange(0, 29, 1))

fig2 = plt.figure()
search_spaces = [fig2.add_subplot(131)]

paths = [[], []]
col = cm.hot.copy()
# search_spaces[0].set_xticks([])
# search_spaces[0].set_yticks([])
# search_spaces[0].grid(True, zorder=0, color="white")
# search_spaces[0].set_facecolor('#eaeaf2')
search_spaces[0].axis('off')
col.set_bad("#00000000")
search_spaces[0].set_title("Shekel", fontsize=30)

search_spaces[0].imshow(m1, cmap=col, alpha=0.75, zorder=5)
search_spaces = [fig2.add_subplot(132)]
search_spaces[0].axis('off')
col.set_bad("#00000000")
search_spaces[0].set_title("Rosenbrock", fontsize=30)

search_spaces[0].imshow(m2, cmap=col, alpha=0.75, zorder=5)
search_spaces = [fig2.add_subplot(133)]
search_spaces[0].axis('off')
col.set_bad("#00000000")
search_spaces[0].set_title("Himmelblau", fontsize=30)

im3 = search_spaces[0].imshow(m3, cmap=col, alpha=0.75, zorder=5)

fig2.subplots_adjust(right=0.8)
cbar_ax = fig2.add_axes([0.9, 0.1, 0.02, 0.8])
fig2.colorbar(im3, cax=cbar_ax)
cbar_ax.tick_params(labelsize=30)

plt.show(block=True)

def paint_paths(index_=0, face_=0):
    global paths

    [lines.remove() for lines in paths[face_]]
    paths[face_] = []
    current_data = m_vect2[ind[index_][1:ind[index_][0] + 1] - 1]

    paths[face_].append(search_spaces[face_].quiver(current_data[:-1, 0], current_data[:-1, 1],
                                                    current_data[1:, 0] - current_data[:-1, 0],
                                                    -current_data[1:, 1] + current_data[:-1, 1],
                                                    label="ASV1", color='r', alpha=0.9, zorder=6))
    if number_of_ships > 1:
        current_data = m_vect2[ind[index_][ind[index_][0] + 2:ind[index_][0] + ind[index_][ind[index_][0] + 1] + 2] - 1]
        paths[face_].append(search_spaces[face_].quiver(current_data[:-1, 0], current_data[:-1, 1],
                                                        current_data[1:, 0] - current_data[:-1, 0],
                                                        -current_data[1:, 1] + current_data[:-1, 1], label="ASV2",
                                                        color='g', alpha=0.9, zorder=6))
    if number_of_ships > 2:
        current_data = m_vect2[ind[index_][ind[index_][0] + ind[index_][ind[index_][0] + 1] + 3:] - 1]
        paths[face_].append(search_spaces[face_].quiver(current_data[:-1, 0], current_data[:-1, 1],
                                                        current_data[1:, 0] - current_data[:-1, 0],
                                                        -current_data[1:, 1] + current_data[:-1, 1], label="ASV3",
                                                        color='b', alpha=0.9, zorder=6))


paint_paths()
l = search_spaces[0].legend(fancybox=True, title="Paths for Best Individual", prop={"size": 20})
l.get_title().set_fontsize('14')

plt.show()
