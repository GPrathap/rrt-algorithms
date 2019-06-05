#
#
#
import numpy as np
# from src.utilities.plotting_cpp import Plot
#
# import numpy
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


#
# # Returns theta in [-pi/2, 3pi/2]
# def generate_theta(a, b):
#     u = random.random() / 4.0
#     theta = numpy.arctan(b/a * numpy.tan(2*numpy.pi*u))
#
#     v = random.random()
#     if v < 0.25:
#         return theta
#     elif v < 0.5:
#         return numpy.pi - theta
#     elif v < 0.75:
#         return numpy.pi + theta
#     else:
#         return -theta
#
# def radius(a, b, theta):
#     return a * b / numpy.sqrt((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)
#
# def random_point(a, b):
#     random_theta = generate_theta(a, b)
#     max_radius = radius(a, b, random_theta)
#     random_radius = max_radius * numpy.sqrt(random.random())
#
#     return numpy.array([
#         random_radius * numpy.cos(random_theta),
#         random_radius * numpy.sin(random_theta)
#     ])
#
# a = 2
# b = 1
#
# points = numpy.array([random_point(a, b) for _ in range(2000)])
#
# plt.scatter(points[:,0], points[:,1])
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

    # generate points
    pt = np.random.normal(0, 1, [npts, ndims]);

    # get scalings for each point onto the surface of a unit hypersphere
    fac = np.sum(pt ** 2, axis=1)

    # calculate scaling for each point to be within the unit hypersphere
    # with radii rs
    fac = (rs ** (1.0 / ndims)) / np.sqrt(fac)

    pnts = np.zeros((npts, ndims));

    # scale points to the ellipsoid using the eigenvalues and rotate with
    # the eigenvectors and add centroid
    d = np.sqrt(np.diag(e))
    d.shape = (ndims, 1)

    for i in range(0, npts):
        # scale points to a uniform distribution within unit hypersphere
        pnts[i, :] = fac[i] * pt[i, :]
        pnts[i, :] = np.dot(np.multiply(pnts[i, :], np.transpose(d)), np.transpose(v)) + cent

    return pnts



covmat = np.diag((7, 4, 4))
pnts = draw_from_ellipsoid(covmat, 0, 100000)
plt.scatter(pnts[:,0], pnts[:,1], pnts[:,2])
plt.show()


