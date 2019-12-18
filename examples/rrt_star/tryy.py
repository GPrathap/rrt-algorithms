import pandas as pd


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
plt.rcParams.update({'font.size': 35})


df = pd.read_csv('/home/geesara/Downloads/datar.csv')

dd = df.iloc[:,0]
item1 = []
for f in dd:
    item1.append(f)

dd = df.iloc[:,1]
item2 = []
for f in dd:
    item2.append(f)
dd = df.iloc[:,2]
item3 = []
for f in dd:
    item3.append(f)
dd = df.iloc[:,3]
item4 = []
for f in dd:
    item4.append(f)

item1 = np.array(item1)
item2 = np.array(item2)
item3 = np.array(item3)
item4 = np.array(item4)


print(item3)


time = np.arange(11)
temp = item3
Swdown = item2
Rn = item4

fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(time, Swdown, '-', label = 'RC')
lns2 = ax.plot(time, Rn, '-', label = 'OC')
ax2 = ax.twinx()
lns3 = ax2.plot(time, temp, '-r', label = 'CC')

# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("Robot count")
ax.set_ylabel(r"RC,OC")
ax2.set_ylabel(r"CC")
ax2.set_ylim(0, 0.4)
ax.set_ylim(3.0,3.4)
plt.show()

#
#
#
