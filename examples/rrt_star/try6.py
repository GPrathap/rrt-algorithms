


import numpy as np
from src.utilities.plotting_cpp import Plot
import matplotlib.pyplot as plt

import pandas as pd
#/home/geesara/project/rrt-algorithms/start_and_end_pose.npy


index = 0

path_rrt = np.load('/home/geesara/Desktop/trajectory_planning_pose.csv')

path_rrt_ori = np.load('/home/geesara/Desktop/trajectory_original_pose.csv')


path_rrt_ori = path_rrt_ori.reshape(-1, 3)

# trees = np.load('/dataset/edges.npy')
# path_rrt = np.load('/dataset/rrt_path.npy')
# path_bspline = np.load('/dataset/rrt_path_modified.npy')
# obstacles = np.load('/dataset/obstacles.npy')
# start_and_end_pose = np.load('/dataset/start_and_end_pose.npy')
# path_catmull = np.load('/dataset/rrt_path_modified_catmull.npy')


# index = 134
#
# trees = np.load('/dataset/rrt/'+str(index)+'_edges.npy')
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
# plot.plot_tree(trees)
#
# plot.plot_path(path_rrt, "orange")

path_rrt = path_rrt.reshape(-1, 9)

paths = pd.read_csv("/home/geesara/project/quadrotor/traj_planning/path.csv", header=None)

# Read data from file 'filename.csv'
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later)
data = pd.read_csv("/home/geesara/project/quadrotor/traj_planning/fff", header=None)
# path_rrt = np.array(data)

plot.plot_trajectory_with_path(path_rrt, path_rrt_ori, "red")
# # plot.plot_path(path_catmull, "green")
# plot.plot_obstacles(obstacles)
# plot.plot_start(start_and_end_pose[0:3])
# plot.plot_goal(start_and_end_pose[3:6])
plot.draw(auto_open=True)


plt.subplot(411)
plt.plot(path_rrt[:,3])
plt.ylabel('X ')
plt.subplot(412)
plt.plot(path_rrt[:,4])
plt.ylabel('Y ')
plt.subplot(413)
plt.plot(path_rrt[:,5])
plt.ylabel('Z ')
plt.subplot(414)

ff = (path_rrt[:,3]**2 + path_rrt[:,4]**2 + path_rrt[:,5]**2)
fff = np.sqrt(ff)

plt.plot(fff)
plt.ylabel('overall')
plt.show()



