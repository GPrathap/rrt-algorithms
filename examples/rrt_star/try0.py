


import numpy as np
from src.utilities.plotting_cpp import Plot
#
#/home/geesara/project/rrt-algorithms/start_and_end_pose.npy
trees = np.load('/dataset/edges.npy')
path_rrt = np.load('/dataset/rrt_path.npy')
path_bspline = np.load('/dataset/rrt_path_modified.npy')
obstacles = np.load('/dataset/obstacles.npy')
start_and_end_pose = np.load('/dataset/start_and_end_pose.npy')
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

# plot
plot = Plot("rrt_star_3d")
plot.plot_tree(trees)

plot.plot_path(path_rrt, "orange")
# plot.plot_path(path_bspline, "red")
# plot.plot_path(path_catmull, "green")
plot.plot_obstacles(obstacles)
plot.plot_start(start_and_end_pose[0:3])
plot.plot_goal(start_and_end_pose[3:6])
plot.draw(auto_open=True)