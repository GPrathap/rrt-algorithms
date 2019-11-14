# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import plotly as py
from plotly import graph_objs as go
from numpy import sin, cos, pi
import numpy as np
colors = ['darkblue', 'teal']
import numpy.linalg as linalg


class Plot(object):
    def __init__(self, filename):
        """
        Create a plot
        :param filename: filename
        """
        self.filename = "./" + filename + ".html"
        self.data = []
        self.layout = {'title': 'Plot',
                       'showlegend': False,

                       }

        self.layout = {}
        self.layout['title'] = 'Chat Times (UTC)'

        self.fig = {'data': self.data,
                    'layout': self.layout}

    def plot_tree(self, trees):
            self.plot_tree_3d(trees)


    def plot_tree_3d(self, trees):
        """
        Plot 3D trees
        :param trees: trees to plot
        """
        i = 0
        for tree in trees:
            for s_d in tree:
                if s_d[0] is not -1:
                    trace = go.Scatter3d(
                        x=[s_d[0], s_d[3]],
                        y=[s_d[1], s_d[4]],
                        z=[s_d[2], s_d[5]],
                        showlegend=False,
                        line=dict(
                            color='#839192'
                        ),
                        mode="lines"
                    )
                    self.data.append(trace)

    def plot_obstacles(self, obstacles):
        """
        Plot obstacles
        :param X: Search Space
        :param O: list of obstaclesCannot plot in
        """
        for obstacle in obstacles:
            for O_i in obstacle:
                obs = go.Mesh3d(
                    x=[O_i[0], O_i[0], O_i[3], O_i[3], O_i[0], O_i[0], O_i[3], O_i[3]],
                    y=[O_i[1], O_i[4], O_i[4], O_i[1], O_i[1], O_i[4], O_i[4], O_i[1]],
                    z=[O_i[2], O_i[2], O_i[2], O_i[2], O_i[5], O_i[5], O_i[5], O_i[5]],
                    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    color='#A6ACAF',
                    opacity=0.90,
                    name="Obstacles",
                )
                self.data.append(obs)


    def draw_ellipsoid_only(self, start_end):
        # phi = np.linspace(0, 2 * pi)
        # theta = np.linspace(-pi / 2, pi / 2)
        # phi, theta = np.meshgrid(phi, theta)
        #
        #
        dis = np.abs(start_end[0] - start_end[4])
        # x = (start_end[0] + start_end[3])/2 + cos(theta) * sin(phi) * dis
        # y = (start_end[1] + start_end[4])/2 + cos(theta) * cos(phi) * 40
        # z = (start_end[2] + start_end[5])/2 + sin(theta)*40

        # your ellispsoid and center in matrix form
        A1 = np.array([[dis, 0, 0], [0, 20, 0], [0, 0, 20]])
        center = [(start_end[0] + start_end[3])/2, (start_end[1] + start_end[4])/2  , (start_end[2] + start_end[5])/2 ]

        A = np.array([-1, 0, 0])
        B = np.array([start_end[3] - start_end[0], start_end[4] - start_end[1], start_end[5] - start_end[2]])




        bx = start_end[3] - start_end[0]
        by = start_end[4] - start_end[1]
        bz = start_end[5] - start_end[2]



        # a = A / (linalg.norm(A))
        a = A / (linalg.norm(A))
        b = B / (linalg.norm(B))
        v =  np.cross(a, b)
        s = linalg.norm(v)
        c = a.dot(b)
        vx = np.array([[0, -v[2], v[1]],
                     [v[2], 0, v[0]],
                     [v[1], v[0], 0]])
        r = np.identity(3, dtype=float)

        r = r + vx + vx*vx*((1-c)/np.sqrt(s))

        rotation = r


        # find the rotation matrix and radii of the axes
        # U, s, rotation1 = linalg.svd(A1)
        # radii = 1.0 / np.sqrt(s)

        # now carry on with EOL's answer
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        x = (dis/2.5) * np.outer(np.cos(u), np.sin(v))
        y = 50 * np.outer(np.sin(u), np.sin(v))
        z = 50 * np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

        # obs = go.Mesh3d({ 'x': x.flatten(), 'y': y.flatten(), 'z': z.flatten(), 'alphahull': 0})
        surface = go.Surface(x=x, y=y, z=z, showscale=False, opacity=0.7,  name="Search space")

        self.data.append(surface)

    def draw_ellipsoid(self, start_end):
        # phi = np.linspace(0, 2 * pi)
        # theta = np.linspace(-pi / 2, pi / 2)
        # phi, theta = np.meshgrid(phi, theta)
        #
        #
        dis = np.abs(start_end[0] - start_end[4])
        # x = (start_end[0] + start_end[3])/2 + cos(theta) * sin(phi) * dis
        # y = (start_end[1] + start_end[4])/2 + cos(theta) * cos(phi) * 40
        # z = (start_end[2] + start_end[5])/2 + sin(theta)*40

        # your ellispsoid and center in matrix form
        A1 = np.array([[dis, 0, 0], [0, 20, 0], [0, 0, 20]])
        center = [(start_end[0] + start_end[3])/2, (start_end[1] + start_end[4])/2  , (start_end[2] + start_end[5])/2 ]

        A = np.array([-1, 0, 0])
        B = np.array([start_end[3] - start_end[0], start_end[4] - start_end[1], start_end[5] - start_end[2]])




        bx = start_end[3] - start_end[0]
        by = start_end[4] - start_end[1]
        bz = start_end[5] - start_end[2]



        # a = A / (linalg.norm(A))
        a = A / (linalg.norm(A))
        b = B / (linalg.norm(B))
        v =  np.cross(a, b)
        s = linalg.norm(v)
        c = a.dot(b)
        vx = np.array([[0, -v[2], v[1]],
                     [v[2], 0, v[0]],
                     [v[1], v[0], 0]])
        r = np.identity(3, dtype=float)

        r = r + vx + vx*vx*((1-c)/np.sqrt(s))

        rotation = r


        # find the rotation matrix and radii of the axes
        # U, s, rotation1 = linalg.svd(A1)
        # radii = 1.0 / np.sqrt(s)

        # now carry on with EOL's answer
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        x = (dis/2.5) * np.outer(np.cos(u), np.sin(v))
        y = 20 * np.outer(np.sin(u), np.sin(v))
        z = 20 * np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

        # obs = go.Mesh3d({ 'x': x.flatten(), 'y': y.flatten(), 'z': z.flatten(), 'alphahull': 0})
        surface = go.Surface(x=x, y=y, z=z, showscale=False, opacity=0.3,  name="Search space")

        self.data.append(surface)

    def plot_path(self, path, color, name, width, mode):
        """
        Plot path through Search Space
        :param X: Search Space
        :param path: path through space given as a sequence of points
        """
        x, y, z = [], [], []
        for i in path:
            print(i)
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            name=name,
            line=dict(
                color=color,
                width=width,
            ),
            mode=mode
        )

        self.data.append(trace)

    def plot_path_dash(self, path, color, name, width, mode):
        """
        Plot path through Search Space
        :param X: Search Space
        :param path: path through space given as a sequence of points
        """
        x, y, z = [], [], []
        for i in path:
            print(i)
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            name=name,
            line=dict(
                color=color,
                width=width,
                dash="dot",
            ),
            mode=mode
        )

        self.data.append(trace)

    def plot_trajectory(self, path, color):
        """
        Plot path through Search Space
        :param X: Search Space
        :param path: path through space given as a sequence of points
        """
        x, y, z = [], [], []
        for i in path:
            # print(i)
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            line=dict(
                color=color,
                width=8
            ),
            mode="markers"
        )
        self.data.append(trace)
        trace = go.Scatter3d(
            x=x*3,
            y=y,
            z=z,
            line=dict(
                color="orange",
                width=8
            ),
            mode="lines"
        )


        self.data.append(trace)

    def plot_trajectory_with_path(self, path1, path2, color):
        """
        Plot path through Search Space
        :param X: Search Space
        :param path: path through space given as a sequence of points
        """
        x, y, z = [], [], []
        for i in path1:
            # print(i)
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            line=dict(
                color="red",
                width=8
            ),
            mode="lines"
        )
        self.data.append(trace)

        x, y, z = [], [], []
        for i in path2:
            # print(i)
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])

        trace = go.Scatter3d(
            x=x * 3,
            y=y,
            z=z,
            line=dict(
                color="orange",
                width=8
            ),
            mode="lines"
        )

        self.data.append(trace)

    def plot_start(self, x_init, color, name):
        """
        Plot starting point
        :param X: Search Space
        :param x_init: starting location
        """
        trace = go.Scatter3d(
            x=[x_init[0]],
            y=[x_init[1]],
            z=[x_init[2]],
            line=dict(
                color=color,
                width=10
            ),
            name=name,
            mode="markers"
        )

        self.data.append(trace)


    def plot_goal(self, x_goal, color, name):
        """
        Plot goal point
        :param X: Search Space
        :param x_goal: goal location
        """
        trace = go.Scatter3d(
            x=[x_goal[0]],
            y=[x_goal[1]],
            z=[x_goal[2]],
            line=dict(
                color=color,
                width=10
            ),
            name=name,
            showlegend=True,
            mode="markers"
        )

        self.data.append(trace)

    def draw(self, auto_open=True):
        """
        Render the plot to a file
        """
        # py.offline.plot(self.fig, filename=self.filename, auto_open=auto_open)
        fig = go.Figure(
            data= self.fig['data'],
            layout=go.Layout( xaxis=go.layout.XAxis(
            # title=go.layout.xaxis.Title(
            #     text="x Axis",
            #     font=dict(
            #         family="Courier New, monospace",
            #         size=4788,
            #         color="#7f7f7f"
            #     )
            # )
        ),
        # yaxis=go.layout.YAxis(
        #     title=go.layout.yaxis.Title(
        #         text="y Axis",
        #         font=dict(
        #             family="Courier New, monospace",
        #             size=18,
        #             color="#7f7f7f"
        #         )
        #     )
        # )
            )
            )

        # fig.update_layout(
        #     xaxis=go.XAxis(
        #         title='Time',
        #         showticklabels=False),
        #     yaxis=go.YAxis(
        #         title='Age'
        #     )
        # )\
        fig.update_layout(scene=dict(
            xaxis=dict(nticks=4, showgrid=True, title='', tickfont=dict(size=16)),
            yaxis=dict(nticks=4, showgrid=True, title='', tickfont=dict(size=16)),
            zaxis=dict(nticks=4, showgrid=True, title='', tickfont=dict(size=16))),
            showlegend=True,
            margin=dict(r=0, l=0, b=0, t=0),
            # legend_orientation="v",
            width=1000,
            height=800,
            legend=go.layout.Legend(
                x=0,
                y=0,
                # orientation='h',
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=18,
                    color="black"
                ),
                # bgcolor="#D7DBDD",
            )

        )
        fig.update_yaxes(automargin=True)

        fig.show()
