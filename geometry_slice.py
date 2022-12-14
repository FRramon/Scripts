# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:53:50 2022

@author: GALADRIEL_GUEST

This script is a place where the geometrical functions that are used in every scripts are grouped.
"""

#%% Imports

import importlib
import glob
import os as os
import xml.etree.cElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import re
import os
from os import listdir
from os.path import isfile, join
import importlib


import tecplot as tp
from tecplot.exception import *
from tecplot.constant import *
from tecplot.constant import PlotType
from tecplot.constant import PlotType, SliceSource
import scipy
import scipy.io
from scipy.interpolate import interp1d
import logging
import skg
from skg import nsphere
import pickle
from tqdm import tqdm
import random


#%% Functions


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


def find_next(points,i,length):
    """
    

    Parameters
    ----------
    points : np array of 3d points
    i : int, starting index
    length : float, the length that is wanted between too points

    Returns
    -------
    int : index of the next points spaced by a distance 'length' from points[i]

    """
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
    """
    

    Parameters
    ----------
    points : numpy array, set of 3d points
    length : distance wanted between the points

    Returns
    -------
    TYPE
        the index of the points that is at the distance 'length' of the last point in the set 

    """
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



def space_array(fname,length):
    """
    

    Parameters
    ----------
    fname : .pth file of control points
    length :distance wanted between the points

    Returns
    -------
    points_spaced :return a set of points equally spaced by the distance 'length'

    """
    
    points_spline = get_spline_points(fname, 1)
    i=0
    L = []

    while i<points_spline.shape[0]-find_last(points_spline,length):
    
        i+=find_next(points_spline,i,length)
        L.append(i)
        
    points_spaced = np.array([points_spline[j,:] for j in L])

        
    return points_spaced




def calculate_normal_vectors(points):
    """


    Parameters
    ----------
    points : (n,3) array of coordinates

    Returns
    -------
    vectors : (n-1,3) array of vectors : i --> i+1

    """
    n = points.shape[0]
    # n-1 vectors
    vectors = np.zeros((n - 1, 3))
    for i in range(n - 1):
        # substracting i vector from i+1
        vectors[i, 0] = points[i + 1, 0] - points[i, 0]
        vectors[i, 1] = points[i + 1, 1] - points[i, 1]
        vectors[i, 2] = points[i + 1, 2] - points[i, 2]

    return vectors


def calculate_norms(vectors):
    """


    Parameters
    ----------
    vectors : (n,3) array of vectors.

    Returns
    -------
    norms : (n,1) array of  euclidean norms of the vectors in input.

    """
    norms = np.zeros((vectors.shape[0], 1))
    for i in range(norms.shape[0]):
        norm_i = np.linalg.norm(vectors[i, :])
        norms[i] = norm_i
    return norms



def create_dpoint(pinfo, case, length):
    """


    Parameters
    ----------
    pinfo : str, example : 'pt2' , 'vsp7'
    case : str, 'baseline' or 'vasospasm'

    Returns
    -------
    dpoint_i : dict of all the control points for the vessels of the patient

    """

   
    folder = "_segmentation"
    pathpath = (
        "N:/vasospasm/"
        + pinfo
        + "/"
        + case
        + "/1-geometry/"
        + pinfo
        + "_"
        + case
        + folder
        + "/paths"
    )

    os.chdir(pathpath)
    onlyfiles = []
    for file in glob.glob("*.pth"):
        onlyfiles.append(file)
    i = 0
    dpoint_i = {}
    for file in onlyfiles:

        filename = file[:-4]
        dpoint_i["points{}".format(i)] = filename, space_array(file,length)
        i += 1
    return dpoint_i


def find_outlet(dpoints):

    pinfo = "pt7"
    case = "baseline"

    filename = pinfo + "_" + case + ".dat"
    logging.basicConfig(level=logging.DEBUG)

    # Run this script with "-c" to connect to Tecplot 360 on port 7600
    # To enable connections in Tecplot 360, click on:
    #   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

    tp.session.connect()
    tp.new_layout()
    frame = tp.active_frame()

    dir_file = (
        "N:/vasospasm/"
        + pinfo
        + "/"
        + case
        + "/3-computational/hyak_submit/"
        + filename
    )

    data_file_baseline = tp.data.load_fluent(
        case_filenames=[
            "N:/vasospasm/"
            + pinfo
            + "/"
            + case
            + "/3-computational/hyak_submit/"
            + pinfo
            + "_"
            + case
            + ".cas"
        ],
        data_filenames=[dir_file],
    )

    # for i in range(len(dp)):
    #     plot_vessels_outlet(dp,i)
    name = "pt7_baseline.bas"
    points = dp.get("points{}".format(11))[1]
    name_vessel = dp.get("points{}".format(11))[0]

    cx = data_file_baseline.zone(name).values("X")[:]
    cy = data_file_baseline.zone(name).values("Y")[:]
    cz = data_file_baseline.zone(name).values("Z")[:]
    x_base = np.asarray(cx)
    y_base = np.asarray(cy)
    z_base = np.asarray(cz)

    coordinates_circle = np.array([x_base, y_base, z_base]).T

    return coordinates_circle


def find_center(coordinates_circle):

    x_mean = np.mean(coordinates_circle[:, 0])
    y_mean = np.mean(coordinates_circle[:, 1])
    z_mean = np.mean(coordinates_circle[:, 2])

    center = np.array([x_mean, y_mean, z_mean])

    return center


def find_radius(center, coord_center):

    L = []
    for i in range(coord_center.shape[0]):
        L.append(np.linalg.norm(center - coord_center[i, :]))
    radius = max(L)

    Lrad = [L.index(x) for x in L if x > 0.9 * radius]
    coord_rad = np.asarray([coord_center[i, :] for i in Lrad])

    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.grid()
    # ax.scatter(coord_rad[:, 0], coord_rad[:, 1], coord_rad[:, 2], label="radius")

    # ax.plot(center[0], center[1], center[2], c="k", marker="x")

    # Xrad = np.zeros((2, 3))
    # Xrad[0] = center
    # Xrad[1] = coord_rad[1]  # Plot the radius
    # ax.plot(Xrad[:, 0], Xrad[:, 1], Xrad[:, 2])

    # ax.view_init(30, 0)
    # ax.legend()
    # plt.show()

    return radius


# Idea : remove a radius on the intersection for all vessels involved
# Ideally want to remove a radius of the vessel which make the intersection (ex pcom in pca -> p1 & p2)






def crop_ICAs(pinfo,case,points_LICA,points_RICA):
    """
    

    Parameters
    ----------
    pinfo : str: ex 'pt7'
    case : str, ex 'baseline'
    points_LICA : numpy array, set of points of Left ICA
    points_RICA : numpy array, set of points of right ICA

    Returns
    -------
    Updated points_LICA and points_RICA, cropped (the value of the separation point is found inside the function)
    """
    
    os.chdir('N:/vasospasm/' + pinfo + '/' + case+ '/1-geometry')
    
    if 'tica_l.dat' in glob.glob('*.dat'):
        
    
    
        Sides=['L','R']
        POINTS= []
        for side in Sides:
            if side=='L':
                points=points_LICA
                filename='tica_l.dat'
                # sep_point=np.array([-0.003168023657053709, -0.03868599608540535, -0.4955081045627594])
                
            elif side=='R':
                points=points_RICA
                filename='tica_r.dat'
    
                # sep_point=np.array([-0.024270353838801384, -0.036143455654382706, -0.491994708776474])
                
           
            f = open('N:/vasospasm/' + pinfo + '/' + case+ '/1-geometry/' + filename)
            data = np.loadtxt(f,
                          dtype=str,
                          usecols=(0,1,2),
                          delimiter=' ')
        
            L_coord=[float(x) for x in data[1,:]]
            sep_point=np.array(L_coord)
            
            points_ant,points_sup=bifurcation_one(points,sep_point)
            
            POINTS.append(points_sup)
    
    
        return POINTS[0],POINTS[1]
    
    else:
        return points_LICA,points_RICA
    
    




def calculate_norms(vectors):
    """


    Parameters
    ----------
    vectors : (n,3) array of vectors.

    Returns
    -------
    norms : (n,1) array of  euclidean norms of the vectors in input.

    """
    norms = np.zeros((vectors.shape[0], 1))
    for i in range(norms.shape[0]):
        norm_i = np.linalg.norm(vectors[i, :])
        norms[i] = norm_i
    return norms


def find_number_of_steps(points_vessel, radius):
    """
    

    Parameters
    ----------
    points_vessel :set of control points of the vessel
    radius :radius of the intersecting vessel

    Returns
    -------
    step_vessel : number of points that need to be removed

    """

    # Convert the radius into the equivalent of steps in the vessel coordinates array

    vect_vessel = calculate_normal_vectors(points_vessel)
    norms_vessel = calculate_norms(vect_vessel)

    # Compute the norms from 0 to i, i variating between 0 and len(vessel)
    L_dist_along = [np.sum(norms_vessel[0:i]) for i in range(norms_vessel.shape[0])]
    # Compare the previous norms and the radius
    L_compare_r = [abs(L_dist_along[i] - radius/2) for i in range(len(L_dist_along))]
    # Select the index of the minimum distance, which correspond to the indice to remove.
    step_vessel = L_compare_r.index(min(L_compare_r))

    # return step_bas,step_lsc,step_rsc

    return step_vessel

def bifurcation_and_radius_remove(points_to_divide,points_bifurc,center_bifurc):
    """
    This function separate a set of points in two, at the bifurcation point. There is also a truncation
    which is made, by removing the radius of the bifurcating vessel on each side.

    Parameters
    ----------
    points_to_divide : set of points
    points_bifurc : set of points of the intersecting vessel
    center_bifurc : center and radius of the intersecting vessel, at the beginning and the end of the vessle

    Returns
    -------
    points_1 : numpy array, result of the first separation at the bifurcation point
    points_2 : numpy array, result of the first separation at the bifurcation point.

    """
    
    nb_norms_start = []
    nb_norms_end = []
    target= [points_bifurc[0], points_bifurc[points_bifurc.shape[0] - 1]]
    for i in range(points_to_divide.shape[0]):
        norm_start = np.linalg.norm(target[0] - points_to_divide[i])
        norm_end = np.linalg.norm(target[1] - points_to_divide[i])

        nb_norms_start.append(norm_start)
        nb_norms_end.append(norm_end)

    Ltot_norms = nb_norms_end + nb_norms_start
    lmini = np.min(Ltot_norms)
    limin = Ltot_norms.index(lmini)

    if limin > len(nb_norms_end):
        limin_final = limin - len(nb_norms_end)
    else:
        limin_final = limin

    points_1 = points_to_divide[:limin_final]
    points_2 = points_to_divide[limin_final:]
    
  
        
    if limin <= len(nb_norms_end):
        indice_1 = find_number_of_steps(
            points_1, center_bifurc.get("center1")[1]
        )
        indice_2 = find_number_of_steps(
            points_2, center_bifurc.get("center1")[1]
        )
    else:
        indice_1 = find_number_of_steps(
            points_1, center_bifurc.get("center2")[1]
        )
        indice_2 = find_number_of_steps(
            points_2, center_bifurc.get("center2")[1]
        )

    points_1 = points_1[: points_1.shape[0] - indice_1]
    points_2 = points_2[indice_2:]


    return points_1, points_2


def bifurcation(points_to_divide,points_bifurc):
    """
    This function make the separation of a vessel with another vessel. There is no
    truncation

    Parameters
    ----------
    points_to_divide : numpy array, set of control points that need to be divided
    points_bifurc : numpy array, set of control points of the vessel that bifurcate with the first set of points

    Returns
    -------
    points_1 : first part of the divided set of points
    points_2 : second part of the divided set of points
    case_center : 1 if the end of the bifurcation vessel is in fact at the last point, 
    2 if it is reversed and so the end of the vessel is at the first point.

    """
    
    nb_norms_start = []
    nb_norms_end = []
    target= [points_bifurc[0], points_bifurc[points_bifurc.shape[0] - 1]]
    for i in range(points_to_divide.shape[0]):
        norm_start = np.linalg.norm(target[0] - points_to_divide[i])
        norm_end = np.linalg.norm(target[1] - points_to_divide[i])

        nb_norms_start.append(norm_start)
        nb_norms_end.append(norm_end)

    Ltot_norms = nb_norms_end + nb_norms_start
    lmini = np.min(Ltot_norms)
    limin = Ltot_norms.index(lmini)

    if limin > len(nb_norms_end):
        limin_final = limin - len(nb_norms_end)
    else:
        limin_final = limin

    points_1 = points_to_divide[:limin_final]
    points_2 = points_to_divide[limin_final:]
    
    if limin<len(nb_norms_end):
        case_center=1
    else:
        case_center=2
    

    return points_1, points_2,case_center


def bifurcation_one(points_to_divide,coord_bifurc):
    """
    

    Parameters
    ----------
    points_to_divide :set of points to divide
    coord_bifurc : coordinate of the point that is used to divide the set of points

    Returns
    -------
    points_1 : first part of the set of points
    points_2 : second part of the set of points

    """
    
    nb_norms_start = []
   
    for i in range(points_to_divide.shape[0]):
        norm_start = np.linalg.norm(coord_bifurc - points_to_divide[i])
      

        nb_norms_start.append(norm_start)
       
    lmini = np.min(nb_norms_start)
    limin =nb_norms_start.index(lmini)
  

    points_1 = points_to_divide[:limin]
    points_2 = points_to_divide[limin:]
    

    return points_1, points_2

def remove_center(points_1,points_2,center_bifurc,case_center):
    """
    

    Parameters
    ----------
    points_1 : TYPE
        DESCRIPTION.
    points_2 : TYPE
        DESCRIPTION.
    center_bifurc : TYPE
        DESCRIPTION.
    case_center : TYPE
        DESCRIPTION.

    Returns
    -------
    points_1 : TYPE
        DESCRIPTION.
    points_2 : TYPE
        DESCRIPTION.

    """
    
    if case_center==1:
        indice_1 = find_number_of_steps(
            points_1, center_bifurc.get("center1")[1]
        )
        indice_2 = find_number_of_steps(
            points_2, center_bifurc.get("center1")[1]
        )
    else:
        indice_1 = find_number_of_steps(
            points_1, center_bifurc.get("center2")[1]
        )
        indice_2 = find_number_of_steps(
            points_2, center_bifurc.get("center2")[1]
        )

    points_1 = points_1[indice_1:]
    points_2 = points_2[indice_2:]
    
    return points_1,points_2


def get_center_radius(fname, pinfo, case):

    os.chdir(
        "N:/vasospasm/"
        + pinfo
        + "/"
        + case
        + "/1-geometry/"
        + pinfo
        + "_"
        + case
        + "_segmentation/Segmentations"
    )

    with open(fname) as f:
        xml = f.read()
        root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    # find the branch of the tree which contains control points

     # index depends on the coding version of the xml file. Some have a format branch, other don't
    index=0
    if 'format' in str(root[0]):
        index=1

    branch = root[index][0]
    n_points = len(branch)
    dsurfaces = {}
    for j in range(1, n_points):
        s_points = np.zeros((len(branch[j][2]), 3))
        for i in range(0, len(branch[j][2])):
            leaf = branch[j][2][i].attrib

            s_points[i][0] = float(leaf.get("x")) * 0.001
            s_points[i][1] = float(leaf.get("y")) * 0.001
            s_points[i][2] = float(leaf.get("z")) * 0.001

        dsurfaces["surface{}".format(j)] = s_points

    dcenter = {}
    for j in range(1, len(dsurfaces) + 1):

        L = np.asarray(dsurfaces.get("surface{}".format(j)))

        center_x = np.mean(L[:, 0])
        center_y = np.mean(L[:, 1])
        center_z = np.mean(L[:, 2])

        center = np.array((center_x, center_y, center_z))

        Lradius = []
        for i in range(L.shape[0]):
            Lradius.append(np.linalg.norm(center - L[i, :]))
        radius = max(Lradius)

        dcenter["center{}".format(j)] = center, radius

    return dcenter

def get_center_radius_ulti(fname, pinfo, case):

    os.chdir(
        "N:/vasospasm/"
        + pinfo
        + "/"
        + case
        + "/1-geometry/"
        + pinfo
        + "_"
        + case
        + "_segmentation/Segmentations"
    )

    fname = "L_ICA_MCA.ctgr"
    with open(fname) as f:
        xml = f.read()
        root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    # find the branch of the tree which contains control points
    index=0
    if 'format' in str(root[0]):
        index=1
    branch = root[index][0]
    n_points = len(branch)
    dsurfaces = {}
    for j in range(1, n_points):
        s_points = np.zeros((len(branch[j][2]), 3))
        for i in range(0, len(branch[j][2])):
            leaf = branch[j][2][i].attrib

            s_points[i][0] = float(leaf.get("x")) * 0.001
            s_points[i][1] = float(leaf.get("y")) * 0.001
            s_points[i][2] = float(leaf.get("z")) * 0.001

        dsurfaces["surface{}".format(j)] = s_points

    dcenter = {}

    Lstart = np.asarray(dsurfaces.get("surface{}".format(1)))
    Lend = np.asarray(dsurfaces.get("surface{}".format(len(dsurfaces))))

    center_x1 = np.mean(Lstart[:, 0])
    center_y1 = np.mean(Lstart[:, 1])
    center_z1 = np.mean(Lstart[:, 2])

    center_x2 = np.mean(Lend[:, 0])
    center_y2 = np.mean(Lend[:, 1])
    center_z2 = np.mean(Lend[:, 2])

    center1 = np.array((center_x1, center_y1, center_z1))
    center2 = np.array((center_x2, center_y2, center_z2))

    Lradius = []
    for i in range(Lstart.shape[0]):
        Lradius.append(np.linalg.norm(center1 - Lstart[i, :]))
    radius1 = max(Lradius)

    Lradius = []
    for i in range(Lend.shape[0]):
        Lradius.append(np.linalg.norm(center2 - Lend[i, :]))
    radius2 = max(Lradius)

    dcenter["center{}".format(1)] = center1, radius1
    dcenter["center{}".format(2)] = center2, radius2

    return dcenter
