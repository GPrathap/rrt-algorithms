# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from src.utilities.obstacle_generation import generate_random_obstacles
from src.utilities.plotting import Plot

X_dimensions = np.array([(0, 300), (0, 300), (0, 300)])  # dimensions of Search Space
x_init = (0, 0, 0)  # starting location
x_goal = (300, 300, 300)  # goal location

Q = np.array([(8, 4)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 10000  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions)
n = 0
Obstacles = generate_random_obstacles(X, x_init, x_goal, n)
# create rrt_search
rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
path = rrt.rrt_star()

# plot
plot = Plot("rrt_star_3d_with_random_obstacles")
plot.plot_tree(X, rrt.trees)

num_true_pts = 100
if path is not None:

    path_np = np.array(path)
    plot.plot_path(X, path_np)

    from scipy.interpolate import splprep, splev
    tck, u = splprep([path_np[:,0], path_np[:,1], path_np[:,2]], s=1,  k=2)
    x_knots, y_knots, z_knots = splev(tck[0], tck)
    u_fine = np.linspace(0, 1, num_true_pts)
    new_path = []
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    new_path.append(x_fine)
    new_path.append(y_fine)
    new_path.append(z_fine)

    path_np = np.array(new_path)
    plot.plot_path(X, path_np)


plot.plot_obstacles(X, Obstacles)

plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
