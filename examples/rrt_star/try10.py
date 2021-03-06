


import numpy as np
from src.utilities.plotting_cpp import Plot
import matplotlib.pyplot as plt
from numpy import sin, cos, pi

import pandas as pd
#/home/geesara/project/rrt-algorithms/start_and_end_pose.npy


index = 0

# path_rrt = np.load('/home/geesara/Desktop/trajectory_planning_pose.csv')
#
# path_rrt_ori = np.load('/home/geesara/Desktop/trajectory_original_pose.csv')


# path_rrt_ori = path_rrt_ori.reshape(-1, 3)

# location  = "/home/geesara/Desktop/searchspace/rrt/"
location  = "/home/geesara/Desktop/testdata/"
# location  = "/home/geesara/Desktop/cool/"
index = 45

dataset_loc = location + str(index) + "_"
trees = np.load( dataset_loc + 'edges.npy')
path_astar = np.load( location + 'a_star_path.npy')
path_rrt  = np.load( dataset_loc + 'rrt_star_path.npy')
search_space  = np.load( dataset_loc + 'rrt_search_space.npy')
path_rrt_dy  = np.load( dataset_loc + 'rrt_star_dynamics_path.npy')
# path_bspline = np.load('/dataset/rrt_path_modified.npy')
obstacles = np.load( dataset_loc + 'obstacles.npy')
# obstacles = np.load('/home/geesara/Desktop/searchspace/100_search_space.npy')
start_and_end_pose = np.load(dataset_loc + 'start_and_end_pose.npy')

# state_pose =  np.load(dataset_loc +  'state_vector.npy')[0]
# path_modified = np.load(dataset_loc +  'rrt_path_modified.npy')

# index = 134
#
# trees = np.load(dataset_loc + 'edges.npy')
# path_rrt = np.load('/dataset/rrt/'+str(index)+'_rrt_star_path.npy')
# # path_bspline = np.load('/dataset/rrt_path_modified.npy')
# # path_catmull = np.load('/dataset/rrt_path_modified_catmull.npy')
# obstacles = np.load('/dataset/rrt/'+str(index)+'_obstacles.npy')
# start_and_end_pose = np.load('/dataset/rrt/'+str(index)+'_start_and_end_pose.npy')


# t1, t2, t3 = path[1][0], path[1][1], path[1][2]
# path[1] = path[2]
# path[2] = [t1, t2, t3]
#
# # plot
plot = Plot("rrt_star_3d")
plot.plot_tree(trees)
# #
# path_rrt = path_rrt.reshape(-1, 9)

# paths = pd.read_csv("/home/geesara/project/quadrotor/traj_planning/path.csv", header=None)

# read data from file 'filename.csv'
# (in the same directory that your python process is based)
# control delimiters, rows, column names with read_csv (see later)
# data = pd.read_csv("/home/geesara/project/quadrotor/traj_planning/fff", header=None)
# path_rrt = np.array(data)
#
# plot.plot_trajectory_with_path(path_rrt, path_rrt_ori, "red")
plot.plot_path(path_rrt, "orange", "Improved RRT*", 10, "lines+markers")
# # plot.plot_path(path_modified, "green", "Smoothed (Original RRT*)", 10, "lines")
plot.plot_path(path_rrt_dy, "green", "After Applying LQR Smoothing", 10, "lines")
plot.plot_search_space(search_space, "#808000", "Search Space", 10, "lines")

# plot.plot_path(state_pose[:,0:3], "blue", "Quad pose", 10, "lines+markers")

index = 1
data_set_loc = location + str(index) + "_"
# path_rrt = np.load( dataset_loc + 'rrt_star_path.npy')
# path_modified = np.load(dataset_loc + 'rrt_path_modified.npy')

# state_pose1 =  np.load(data_set_loc +  'state_vector.npy')[0]
# plot.plot_path(state_pose1[:,0:3], "red", "Converted", 10, "lines")
# plot.plot_path(path_rrt, "blue", "Improved RRT*", 10, "lines+markers")
# plot.plot_path(path_astar, "red", "A*", 10, "lines+markers")
#
# # plot.plot_path(path_modified, "black", "Smoothed (Improved RRT*)", 10, "lines")
#
plot.plot_obstacles(obstacles)
# trajectory = [start_and_end_pose[0:3]]
# trajectory.append(start_and_end_pose[3:6])
# plot.plot_path_dash(trajectory, "red", "Trajectory", 12, "lines")
plot.plot_start(start_and_end_pose[0:3], "#A569BD", "$X_{start}$")
plot.plot_goal(start_and_end_pose[3:6], "#239B56", "$X_{goal}$")
# plot.draw_ellipsoid(start_and_end_pose)
plot.draw(auto_open=True)
plt.show()



