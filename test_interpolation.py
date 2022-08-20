# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:00:22 2022

@author: GALADRIEL_GUEST
"""

import xml.etree.cElementTree as ET
import re
import numpy as np
import scipy
import matplotlib.pyplot as plt


def xml_to_points(fname):
    """


    Parameters
    ----------
    fname : .pth file containing the coordinates of the control points

    Returns
    -------
    points : (n,3) array of the coordinates of the control points

    """
    with open(fname) as f:
        xml = f.read()
        root = ET.fromstring(
            re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    # find the branch of the tree which contains control points
    branch = root[1][0][0][0]
    n_points = len(branch)
    points = np.zeros((n_points, 3))
    for i in range(n_points):

        leaf = branch[i].attrib
        # Convert in meters - Fluent simulation done in meters
        points[i][0] = float(leaf.get("x")) * 0.001
        points[i][1] = float(leaf.get("y")) * 0.001
        points[i][2] = float(leaf.get("z")) * 0.001

    return points


def get_control_points(fname):

    with open(fname) as f:
        xml = f.read()
        root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    # index depends on the coding version of the xml file. Some have a format branch, other don't
    index=0
    if 'format' in str(root[0]):
        index=1
        
    # find the branch of the tree which contains control points

    branch = root[index][0][0][0]
    print(branch)
   
    n_points = len(branch)

    # if step !=1:
    #     s_points = np.zeros((int(np.ceil(n_s_points))+1, 3))
    # else:
    s_points = np.zeros((n_points, 3))

    for i in range(0, n_points):
        # k = i // step

        leaf = branch[i].attrib
        # Convert in meters - Fluent simulation done in meters
        s_points[i][0] = float(leaf.get("x")) * 0.001
        s_points[i][1] = float(leaf.get("y")) * 0.001
        s_points[i][2] = float(leaf.get("z")) * 0.001

    return s_points


def get_spline_points(fname, step):

    with open(fname) as f:
        xml = f.read()
        root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    # index depends on the coding version of the xml file. Some have a format branch, other don't
    index=0
    if 'format' in str(root[0]):
        index=1
        
    # find the branch of the tree which contains control points

    branch = root[index][0][0][1]
   
    n_points = len(branch)
    n_s_points = n_points / step

    # if step !=1:
    #     s_points = np.zeros((int(np.ceil(n_s_points))+1, 3))
    # else:
    s_points = np.zeros((int(np.ceil(n_s_points)), 3))

    for i in range(0, n_points, step):
        k = i // step

        leaf = branch[i][0].attrib
        # Convert in meters - Fluent simulation done in meters
        s_points[k][0] = float(leaf.get("x")) * 0.001
        s_points[k][1] = float(leaf.get("y")) * 0.001
        s_points[k][2] = float(leaf.get("z")) * 0.001

    return s_points

# fname = 'N:/vasospasm/vsp19/baseline/1-geometry/vsp19_baseline_segmentation/Paths/M_ACA.pth'

# points_test = get_control_points(fname)

# def split(points):
    
#     points_xy = np.array((points[:,0],points[:,1]))
#     points_z = np.array(points[:,2])
    
#     return points_xy,points_z
    
# points_xyT,points_zT = split(points_test)
# points_xy = points_xyT.T
# points_z =points_zT.T
    
# def get_interp_xy(points_xy):
#     X = np.linspace(points_xy[0,0],points_xy[0,-1],100)
#     Y = np.linspace(points_xy[1,0],points_xy[1,-1],100)
#     print(points_xy[0,0],points_xy[0,-1])
#     return np.array((X,Y))

# #xi = get_interp_xy(points_xyT).T
# xi = points_xy






# plt.scatter(xi[:,0],xi[:,1])
    
# interp = scipy.interpolate.griddata(points_xy,points_z,xi,method ='cubic')
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# ax.grid()
# ax.scatter(xi[:,0],xi[:,1],interp)
# ax.view_init(30,30)





# FIRST INTERPOLATION ON X


fname = 'N:/vasospasm/pt2/baseline/1-geometry/pt2_baseline_segmentation/Paths/R_Pcom_PCA.pth'

points_test = get_control_points(fname)
points_spline = get_spline_points(fname, 1)


def find_next(points,i,length):
    k=i
    norm = 0
    x = points[i,:]
    y = points[k,:]
    while norm < length:
        k+=1
        y = points[k,:]
        norm +=np.linalg.norm(x-y)
        

    return k-i

def find_last(points,length):
    i = points.shape[0]-1
    k=i
    norm = 0
    x = points[i,:]
    y = points[k,:]
    while norm < length:
        k-=1
        y = points[k,:]
        norm +=np.linalg.norm(x-y)
        # print(norm)
        

    return i-k



def construct_array(points,length):
    
    i=0
    L = []

    while i<points_spline.shape[0]-find_last(points_spline,length):
    
        i+=find_next(points_spline,i,length)
        L.append(i)
        
    points_spaced = np.array([points_spline[j,:] for j in L])

        
    return points_spaced


length = 0.00
points_spaced = construct_array(points_spline,length)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
ax.grid()
ax.scatter(points_spaced[:,0],points_spaced[:,1],points_spaced[:,2])

for i in range(1,points_spaced.shape[0]):
    print(np.linalg.norm(points_spaced[i,:]-points_spaced[i-1,:]))
    

#%%
    
# def splitX(points):
#     points_x = points[:,0]
#     points_y = points[:,1]
#     values = points[:,2]
    
#     return points_x.T ,points_y.T, values.T

# points_x,points_y,values = splitX(points_test)

# def get_interp_x(points_x):
#     X = np.linspace(np.min(points_x),np.max(points_x),50)
#     Y = np.linspace(np.min(points_y),np.max(points_y),50)
#     Z = np.linspace(np.min(values),np.max(values),50)

    
#     return X,Y,Z

# xi,yi,zi = get_interp_x(points_x)

# interp_y = scipy.interpolate.griddata(points_x,points_y,xi,method = 'linear')


# interp_v1 = scipy.interpolate.interp1d(points_x,values)
# interp_v2 = scipy.interpolate.interp1d(points_y,values)


# interp_v2 = scipy.interpolate.griddata(points_y,values,interp_y,method = 'cubic')
# plt.scatter(xi,interp_y)
# interp_tot = scipy.interpolate.griddata((points_x,points_y,values),values,(xi,yi,zi))
# znew = f(xi,interp_y)


# f = scipy.interpolate.LinearNDInterpolator(points_xy,values)

# test_abs = np.linspace(0,49,50)

# plt.scatter(points_x,points_y,marker ='X')
#plt.plot(xi,interp_v1(xi))
# plt.scatter(xi,interp_y)
# plt.plot(points_spline[:,0],points_spline[:,1],points_spline[:,2])
# # plt.plot(test_abs,xi)
# # plt.plot(test_abs,interp_y)
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# ax.grid()
# # ax.scatter(xi,interp_y,interp_v1)
# # ax.scatter(xi,interp_y,interp_v2)
# # ax.scatter(xi,yi,zi)
# # ax.plot(points_spline[:,0],points_spline[:,1],points_spline[:,2])

# # # ax.scatter(points_x,points_y,values)
# # ax.scatter(points_i[:,0],points_i[:,1])
# ax.view_init(30,30)

