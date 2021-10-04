import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

import mpl_toolkits.mplot3d.art3d

'''
    Indicate the number of ships  using:
        number_of_ships = 1
    Works with 1, 2 and 3 ships for 3 objectives.
    Works with 1 ship for 4 objectives. (
        
    For a functional resulting figure, one module of the matplotlib library (mpl_toolkits.mplot3d.art3d)
    needs to be modified. More information on https://github.com/FedePeralta/EMOPP
    
    Run the script and select any Pareto point in the Objective Space (1st sub-figure), the selected point will graph
    its individual representation in the second sub-figure (Search Space). 
'''
number_of_ships = 1  # 1, 2 or 3

# include_distance = False  # False for plotting the results with 3 objectives
include_distance = True  # True for plotting the results with 4 objectives (including distance)

if include_distance and number_of_ships != 1:
    raise ValueError("Results for 4 objectives were only obtained with 1 ship")
elif not include_distance and not (0 < number_of_ships < 4):
    raise ValueError("Number of ships must be betwen 1 and 3")


def get_clean(_file):
    for nnan in nans:
        _file[nnan[1], nnan[0]] = np.nan
    return np.ma.array(_file, mask=np.isnan(_file))


dist = '_distance' if include_distance else ''
f3 = pd.read_csv(f"Results/fitness_multi_{number_of_ships}ships{dist}.txt", index_col=0)
print(f3)
ind = []
with open(f"Results/ind_multi_{number_of_ships}ships{dist}.txt") as f:
    line = f.readline()
    while line != "":
        full_ind = np.array([int(i) for i in line[line.find(",") + 2:-2].split(',')])
        ind.append(full_ind)
        line = f.readline()

m = pd.read_csv("Data/map.csv", names=np.arange(0, 19, 1)).to_numpy()
m_vect2 = np.flipud(np.asarray(np.where(m == 1)).reshape(2, -1)).T
nans = np.flipud(np.asarray(np.where(m == 0)).reshape(2, -1)).T
m1 = get_clean(pd.read_csv("Data/himmelblau_gt.csv", names=np.arange(0, 19, 1)).to_numpy())
m2 = get_clean(pd.read_csv("Data/rosenbrock_gt.csv", names=np.arange(0, 19, 1)).to_numpy())
m3 = get_clean(pd.read_csv("Data/shekel_gt.csv", names=np.arange(0, 19, 1)).to_numpy())
fig = plt.figure()

obj_space = fig.add_subplot(121, projection='3d', azim=50, elev=11, proj_type='ortho')
obj_space.set_xlabel('$Shekel$', fontsize=15)
obj_space.set_ylabel('$Himmelblau$', fontsize=15)
obj_space.set_zlabel('$Rosenbrock$', fontsize=15)

if include_distance:
    markers = obj_space.scatter(f3["sh"], f3["hb"], f3["rb"], c=np.arange(0, len(f3["sh"])), cmap='jet', picker=True,
                                pickradius=2, s=f3["di"], alpha=0.2)
else:
    obj_space.scatter(np.zeros_like(f3["hb"]), f3["hb"], f3["rb"], c=np.zeros_like(f3["sh"]), cmap='binary_r',
                      alpha=0.05)
    obj_space.scatter(f3["sh"], np.zeros_like(f3["hb"]), f3["rb"], c=np.zeros_like(f3["sh"]), cmap='binary_r',
                      alpha=0.05)
    obj_space.scatter(f3["sh"], f3["hb"], np.zeros_like(f3["hb"]), c=np.zeros_like(f3["sh"]), cmap='binary_r',
                      alpha=0.05)
    markers = obj_space.scatter(f3["sh"], f3["hb"], f3["rb"], c=np.arange(0, len(f3["sh"])), cmap='jet', picker=True,
                                pickradius=2)

index = 0

selected = [obj_space.plot(f3["sh"][index], f3["hb"][index], f3["rb"][index], 'Xk', zorder=600000,
                           label="Solution", markersize=6, alpha=0.7),
            obj_space.plot([f3["sh"][index], f3["sh"][index]], [f3["hb"][index], f3["hb"][index]],
                           [0, f3["rb"][index]],
                           'k--', alpha=0.5, linewidth=2),
            obj_space.plot([0, f3["sh"][index]], [f3["hb"][index], f3["hb"][index]],
                           [f3["rb"][index], f3["rb"][index]],
                           'k--', alpha=0.5, linewidth=2),
            obj_space.plot([f3["sh"][index], f3["sh"][index]], [0, f3["hb"][index]],
                           [f3["rb"][index], f3["rb"][index]],
                           'k--', alpha=0.5, linewidth=2)
            ]

obj_space.legend()

X, Y = np.meshgrid(np.arange(0, 19, 1), np.arange(0, 29, 1))
z1 = np.full_like(m1, 0)
z2 = np.full_like(m1, 5)
z3 = np.full_like(m1, 10)

search_space = fig.add_subplot(122, projection='3d', azim=-80, elev=-153.5, proj_type='ortho')

paths = []
search_space.set_zticks([0, 5, 10])
search_space.set_zticklabels(["$Hb.$", "$Rs.$", "$Sh.$"])
search_space.get_xaxis().set_ticks([])
search_space.get_yaxis().set_ticks([])
search_space.grid(False)

search_space.plot_surface(X - 0.5, Y - 0.5, z1, facecolors=cm.hot(m1, 0.5))
search_space.plot_surface(X - 0.5, Y - 0.5, z2, facecolors=cm.hot(m2, 0.5))
search_space.plot_surface(X - 0.5, Y - 0.5, z3, facecolors=cm.hot(m3, 0.5))
veh_colors = ['r', 'g', 'b']


def paint_paths(index_=0):
    global paths

    [lines.remove() for lines in paths]
    paths = []

    for i in range(number_of_ships):
        if i == 0:
            current_data = m_vect2[ind[index_][1:ind[index_][0] + 1] - 1]
        elif i == 1:
            current_data = m_vect2[
                ind[index_][ind[index_][0] + 2:ind[index_][0] + ind[index_][ind[index_][0] + 1] + 2] - 1]
        else:
            current_data = m_vect2[ind[index_][ind[index_][0] + ind[index_][ind[index_][0] + 1] + 3:] - 1]

        paths.append(
            search_space.quiver(current_data[:-1, 0], current_data[:-1, 1], np.full_like(current_data[:-1, 0], 0),
                                current_data[1:, 0] - current_data[:-1, 0],
                                current_data[1:, 1] - current_data[:-1, 1],
                                np.zeros_like(current_data[:-1, 0]), label=f"V{i + 1}", color=veh_colors[i]))
        paths.append(
            search_space.quiver(current_data[:-1, 0], current_data[:-1, 1], np.full_like(current_data[:-1, 0], 5),
                                current_data[1:, 0] - current_data[:-1, 0],
                                current_data[1:, 1] - current_data[:-1, 1],
                                np.zeros_like(current_data[:-1, 0]), zorder=10, color=veh_colors[i]))
        paths.append(
            search_space.quiver(current_data[:-1, 0], current_data[:-1, 1], np.full_like(current_data[:-1, 0], 10),
                                current_data[1:, 0] - current_data[:-1, 0],
                                current_data[1:, 1] - current_data[:-1, 1],
                                np.zeros_like(current_data[:-1, 0]), zorder=10, color=veh_colors[i]))


paint_paths()
search_space.legend(fancybox=True, title="Individual Representation")


def onclick(event):
    if event.mouseevent.name == "button_press_event" and event.mouseevent.button == 1:
        global selected
        index_ = markers.z_markers_idx[event.ind[0]]

        [selected.pop(0)[0].remove() for _ in range(4)]

        selected.extend([
            obj_space.plot(f3["sh"][index_], f3["hb"][index_], f3["rb"][index_], 'Xk', zorder=600000, markersize=6,
                           alpha=0.7),
            obj_space.plot([f3["sh"][index_], f3["sh"][index_]], [f3["hb"][index_], f3["hb"][index_]],
                           [0, f3["rb"][index_]],
                           'k--', alpha=0.5, linewidth=2),
            obj_space.plot([0, f3["sh"][index_]], [f3["hb"][index_], f3["hb"][index_]],
                           [f3["rb"][index_], f3["rb"][index_]],
                           'k--', alpha=0.5, linewidth=2),
            obj_space.plot([f3["sh"][index_], f3["sh"][index_]], [0, f3["hb"][index_]],
                           [f3["rb"][index_], f3["rb"][index_]],
                           'k--', alpha=0.5, linewidth=2)])
        paint_paths(index_)
        fig.canvas.draw_idle()


cid = fig.canvas.mpl_connect('pick_event', onclick)

plt.show()
