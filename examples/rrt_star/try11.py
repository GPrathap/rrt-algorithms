#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


data = np.array(np.genfromtxt('/home/geesara/Desktop/testdata/time_stamps.txt', dtype=None, delimiter=','))

rrt = np.zeros((10,2))
rrt_modified = np.zeros((10,2))
astar = np.zeros((10,2))

counter = 0
for values in data:

    if(counter >= 10):
        break
    if(values[0]=="rrt"):
        # print(values[3])
        rrt[counter][0] = 44428.6/values[4]
        rrt[counter][1] = values[1]*1000.0
        # rrt[1] += values[2]
        # rrt[2] += values[3]
        counter += 1

    if (values[0] == "rrt_modified"):
        rrt_modified[counter][0] = 44428.6/values[4]
        rrt_modified[counter][1] = values[1]*1000.0
        # rrt_modified[1] += values[2]
        # rrt_modified[2] += values[3]

    if (values[0] == "astar"):
        astar[counter][0] = 44428.6/values[4]
        astar[counter][1] = values[1]*1000.0
        # astar[1] += values[2]
        # astar[2] += values[3]


# rrt = rrt/counter
# rrt_modified = rrt_modified/counter
# astar = astar/counter
cont = 1
for a,r,rt in zip(astar, rrt, rrt_modified):
    # print("\multirow{1}{*}{1} &"+str(a[0])+"&"+str(r[0])+"&"+str(rt[0]))
    # print(a[0], r[0], rt[0])
    # print("&{0:.5f}&{0:.5f}&{0:.5f}".format(a[0], r[0], rt[0]))
    print("\multirow{1}{*}{%d} & %1.2f&%1.2f&%1.2f " % (cont, a[1], r[1], rt[1]))
    print("\hline")
    cont+=1
# print(rrt)
# print(rrt_modified)
# print(astar)