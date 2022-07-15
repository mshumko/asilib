import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos, sqrt, tan, arctan2, arccos

#Internal imports
import lambert_projection

def transformVector(geom, raxis, rot):
    """
    Input:
    geom: single point geometry (vector)
    raxis: rotation axis as a vector (vector)
    ([0][1][2]) = (x,y,z) = (Longitude, Latitude, Down)
    rot: rotation in radian

    Returns:
    Array: a vector that has been transformed
    """
    sr = sin(rot)
    cr = cos(rot)
    omcr = 1.0 - cr
    tf = np.array([
        [cr + raxis[0]**2 * omcr,
        -raxis[2] * sr + raxis[0] * raxis[1] * omcr,
        raxis[1] * sr + raxis[0] * raxis[2] * omcr],
        [raxis[2] * sr + raxis[1] * raxis[0] * omcr,
        cr + raxis[1]**2 * omcr,
        -raxis[0] * sr + raxis[1] * raxis[2] * omcr],
        [-raxis[1] * sr + raxis[2] * raxis[0] * omcr,
        raxis[0] * sr + raxis[2] * raxis[1] * omcr,
        cr + raxis[2]**2 * omcr]])

    ar = np.dot(geom, tf)
    return ar

def sphericalToVector(inp_ar):
    """
    Convert a spherical measurement into a vector in cartesian space
    [0] = x (+) east (-) west
    [1] = y (+) north (-) south
    [2] = z (+) down
    """
    ar = np.array([0.0, 0.0, 0.0])
    ar[0] = -sin(inp_ar[1])
    ar[1] = sin(inp_ar[0]) * cos(inp_ar[1])
    ar[2] = cos(inp_ar[0]) * cos(inp_ar[1])
    return ar

def vectorToGeogr(vect):
    """
    Returns:
    Array with the components [0] longitude, [1] latitude
    """
    ar = np.array([0.0, 0.0])
    ar[0] = np.arctan2(vect[1], vect[2])
    ar[1] = np.arcsin(-vect[0] / np.linalg.norm(vect))
    return ar

def plotPoint(dip):
    """
    Testfunction for converting, transforming and plotting a point
    """
    plt.subplot(111, projection="lmbrt_equ_area_equ_aspect")

    #Convert to radians
    dip_rad = np.radians(dip)

    #Set rotation to azimuth and convert dip to latitude on north-south axis
    rot = dip_rad[0]
    dip_lat = pi/2 - dip_rad[1]
    plt.plot(0, dip_lat, "ro")
    print(dip_lat, rot)

    #Convert the dip into a vector along the north-south axis
    #x = 0, y = dip
    vect = sphericalToVector([0, dip_lat])
    print(vect, np.linalg.norm(vect))

    #Transfrom the dip to its proper azimuth
    tvect = transformVector(vect, [0,0,1], rot)
    print(tvect, np.linalg.norm(tvect))

    #Transform the vector back to geographic coordinates
    geo = vectorToGeogr(tvect)
    print(geo)
    plt.plot(geo[0], geo[1], "bo")

    plt.grid(True)
    plt.show()

datapoint = np.array([90.0,30])
plotPoint(datapoint)
