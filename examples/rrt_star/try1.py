#
#
#
import numpy as np
# from src.utilities.plotting_cpp import Plot
#
# import numpy
import matplotlib.pyplot as plt
import scipy.interpolate as si
#
# points = [[0, 0], [0, 2], [2, 3], [4, 0], [6, 3], [8, 2], [8, 0]];
# points = np.array(points)
# x = points[:,0]
# y = points[:,1]
#
# t = range(len(points))
# ipl_t = np.linspace(0.0, len(points) - 1, 100)
#
# x_tup = si.splrep(t, x, k=3)
# y_tup = si.splrep(t, y, k=3)
#
# x_list = list(x_tup)
# xl = x.tolist()
# x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]
#
# y_list = list(y_tup)
# yl = y.tolist()
# y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]
#
# x_i = si.splev(ipl_t, x_list)
# y_i = si.splev(ipl_t, y_list)
#
# #==============================================================================
# # Plot
# #==============================================================================
#
# fig = plt.figure()
#
# ax = fig.add_subplot(231)
# plt.plot(t, x, '-og')
# plt.plot(ipl_t, x_i, 'r')
# plt.xlim([0.0, max(t)])
# plt.title('Splined x(t)')
#
# ax = fig.add_subplot(232)
# plt.plot(t, y, '-og')
# plt.plot(ipl_t, y_i, 'r')
# plt.xlim([0.0, max(t)])
# plt.title('Splined y(t)')
#
# ax = fig.add_subplot(233)
# plt.plot(x, y, '-og')
# plt.plot(x_i, y_i, 'r')
# plt.xlim([min(x) - 0.3, max(x) + 0.3])
# plt.ylim([min(y) - 0.3, max(y) + 0.3])
# plt.title('Splined f(x(t), y(t))')
#
# ax = fig.add_subplot(234)
# for i in range(7):
#     vec = np.zeros(11)
#     vec[i] = 1.0
#     x_list = list(x_tup)
#     x_list[1] = vec.tolist()
#     x_i = si.splev(ipl_t, x_list)
#     plt.plot(ipl_t, x_i)
# plt.xlim([0.0, max(t)])
# plt.title('Basis splines')
# plt.show()


# phi = np.linspace(0, 2. * np.pi, 40)
# r = 0.5 + np.cos(phi)  # polar coords
# x, y = r * np.cos(phi), r * np.sin(phi)  # convert to cartesian
#
#
# from scipy.interpolate import splprep, splev
# tck, u = splprep([x, y], s=0)
# new_points = splev(u, tck)
#
#
# fig, ax = plt.subplots()
# ax.plot(x, y, 'ro')
# # ax.plot(new_points[0], new_points[1], 'r-')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


# # 3D example
# total_rad = 10
# z_factor = 3
# noise = 0.1
#
# num_true_pts = 200
# s_true = np.linspace(0, total_rad, num_true_pts)
# x_true = np.cos(s_true)
# y_true = np.sin(s_true)
# z_true = s_true/z_factor
#
# num_sample_pts = 80
# s_sample = np.linspace(0, total_rad, num_sample_pts)
# x_sample = np.cos(s_sample) + noise * np.random.randn(num_sample_pts)
# y_sample = np.sin(s_sample) + noise * np.random.randn(num_sample_pts)
# z_sample = s_sample/z_factor + noise * np.random.randn(num_sample_pts)
#
# from scipy.interpolate import splprep, splev
#
# tck, u = splprep([x_sample,y_sample,z_sample], s=2)
# x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
# u_fine = np.linspace(0,1,num_true_pts)
# x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
#
# fig2 = plt.figure(2)
# ax3d = fig2.add_subplot(111, projection='3d')
# ax3d.plot(x_true, y_true, z_true, 'b')
# ax3d.plot(x_sample, y_sample, z_sample, 'r*')
# ax3d.plot(x_knots, y_knots, z_knots, 'go')
# ax3d.plot(x_fine, y_fine, z_fine, 'g')
# fig2.show()
# plt.show()
def draw_from_ellipsoid(covmat, cent, npts):
    # random uniform points within ellipsoid as per: http://www.astro.gla.ac.uk/~matthew/blog/?p=368
    ndims = covmat.shape[0]

    # calculate eigenvalues (e) and eigenvectors (v)
    eigenValues, eigenVectors = np.linalg.eig(covmat)
    idx = (-eigenValues).argsort()[::-1][:ndims]
    e = eigenValues[idx]
    v = eigenVectors[:, idx]
    e = np.diag(e)

    # generate radii of hyperspheres
    rs = np.random.uniform(0, 1, npts)
    # rs = np.arange(npts)
    # generate points
    pt = np.random.normal(0, 1, [npts, ndims])
    # pt = np.arange(npts*ndims).reshape(npts, ndims)

    # get scalings for each point onto the surface of a unit hypersphere
    fac = np.sum(pt ** 2, axis=1)

    # calculate scaling for each point to be within the unit hypersphere
    # with radii rs

    fac = (rs ** (1.0 / ndims)) / np.sqrt(fac)

    pnts = np.zeros((npts, ndims))

    # scale points to the ellipsoid using the eigenvalues and rotate with
    # the eigenvectors and add centroid
    d = np.sqrt(np.diag(e))
    d.shape = (ndims, 1)

    print(v)
    for i in range(0, npts):
        # scale points to a uniform distribution within unit hypersphere
        pnts[i, :] = fac[i] * pt[i, :]
        print(np.multiply(pnts[i, :], np.transpose(d)))
        pnts[i, :] = np.dot(np.multiply(pnts[i, :], np.transpose(d)), np.transpose(v)) + cent


    return pnts



covmat = np.diag((7, 4, 4))
# pnts = draw_from_ellipsoid(covmat, 0, 10000)
pnts = np.load('/home/geesara/Desktop/searchspace/1_search_space.npy')[0]
print(pnts)
plt.scatter(pnts[:,0], pnts[:,1], pnts[:,2])
plt.show()


