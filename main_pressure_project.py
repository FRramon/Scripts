# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:17:42 2022

@author: Francois


"""


# %% Imports


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


# %% Scripts as modules

os.chdir("N:/vasospasm/pressure_pytec_scripts")

import geometry_slice as geom

# %% Functions


# def xml_to_points(fname):
#     """


#     Parameters
#     ----------
#     fname : .pth file containing the coordinates of the control points

#     Returns
#     -------
#     points : (n,3) array of the coordinates of the control points

#     """
#     with open(fname) as f:
#         xml = f.read()
#         root = ET.fromstring(
#             re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

#     # find the branch of the tree which contains control points
#     branch = root[1][0][0][0]
#     n_points = len(branch)
#     points = np.zeros((n_points, 3))
#     for i in range(n_points):

#         leaf = branch[i].attrib
#         # Convert in meters - Fluent simulation done in meters
#         points[i][0] = float(leaf.get("x")) * 0.001
#         points[i][1] = float(leaf.get("y")) * 0.001
#         points[i][2] = float(leaf.get("z")) * 0.001

#     return points


def get_spline_points(fname, step):

    with open(fname) as f:
        xml = f.read()
        root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    # find the branch of the tree which contains control points
    branch = root[1][0][0][1]

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

    # if step !=1:
    #    leaf = branch[n_points-1][0].attrib
    #    print(leaf)
    #    print(int(np.ceil(n_s_points))+1)
    #    # Convert in meters - Fluent simulation done in meters
    #    s_points[int(np.ceil(n_s_points))][0] = float(leaf.get("x")) * 0.001
    #    s_points[int(np.ceil(n_s_points))][1] = float(leaf.get("y")) * 0.001
    #    s_points[int(np.ceil(n_s_points))][2] = float(leaf.get("z")) * 0.001

    return s_points


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


def get_distance_along(i_vessel, i_dat, dpressure, dvectors, dpoints):
    # Change name function?
    """


    Parameters
    ----------
    dvectors : dictionary of the normal vectors in the patient's circle of Willis

    Returns
    -------
    ddist : dictionnary of the distance along the vessels (use to plot the pressure later)

    """
    dnorms = {}
    ddist = {}
    dvectors = {}
    onlydat, indices_dat = get_list_files_dat(pinfo, case, num_cycle)

    # print(type(indices_dat[i_dat]))
    # print(dpressure)
    # print(dpressure.get('{}'.format(indices_dat[1])).get("pressure{}".format(i_vessel))[1][0])

    # Inverse the order of the points if the pressure is not decreasing

    if (
        dpressure.get("{}".format(indices_dat[i_dat])).get(
            "pressure{}".format(i_vessel)
        )[1][0]
        == 1
    ):  # Check if it is to reverse (marked 1, 0 if not)
        array_points = dpoints.get("points{}".format(i_vessel))[1][
            ::-1
        ]  # reverse the array
        dvectors["vectors{}".format(i_vessel)] = dpoints.get(
            "points{}".format(i_vessel)
        )[0], calculate_normal_vectors(
            array_points
        )  # Compute new vectors
    else:
        array_points = dpoints.get("points{}".format(i_vessel))[1]
        dvectors["vectors{}".format(i_vessel)] = dpoints.get(
            "points{}".format(i_vessel)
        )[0], calculate_normal_vectors(array_points)

    dnorms["norms{}".format(i_vessel)] = dvectors.get("vectors{}".format(i_vessel))[
        0
    ], calculate_norms(dvectors.get("vectors{}".format(i_vessel))[1])

    n_norms = dnorms.get("norms{}".format(i_vessel))[1].shape[0]

    dist = np.zeros((n_norms + 1, 1))
    for j in range(1, n_norms + 1):
        dist[j] = dnorms.get("norms{}".format(i_vessel))[1][:j, :].sum()
        ddist["dist{}".format(i_vessel)] = (
            dnorms.get("norms{}".format(i_vessel))[0],
            dist,
        )
    return ddist


def get_distance(vectors):
    norms = calculate_norms(vectors)
    dist = np.zeros((norms.shape[0] + 1, 1))
    for j in range(1, norms.shape[0] + 1):
        dist[j] = norms[:j, :].sum()

    return dist


def calculate_square_norm(vectors):
    """


    Parameters
    ----------
    vectors : (n,3) array of vectors.

    Returns
    -------
    norms : (n,1) array of square euclidean norms of the vectors in input.

    """
    norms = np.zeros((vectors.shape[0], 1))
    for i in range(norms.shape[0]):
        norm = np.linalg.norm(vectors[i, :])
        norms[i] = norm**2
    return norms


# def create_dpoint(pinfo, case):
#     """


#     Parameters
#     ----------
#     pinfo : str, example : 'pt2' , 'vsp7'
#     case : str, 'baseline' or 'vasospasm'

#     Returns
#     -------
#     dpoint_i : dict of all the control points for the vessels of the patient

#     """

#     if pinfo == 'pt2':
#         folder = '_segmentation_no_vti'
#     else:
#         folder = '_segmentation'
#     pathpath = 'N:/vasospasm/' + pinfo + '/' + case+'/1-geometry/' + \
#         pinfo + '_' + case + folder + '/paths'

#     os.chdir(pathpath)
#     onlyfiles = []
#     for file in glob.glob("*.pth"):
#         onlyfiles.append(file)
#     i = 0
#     dpoint_i = {}
#     for file in onlyfiles:
#         filename = file[:-4]
#         dpoint_i["points{}".format(
#             i)] = filename, get_spline_points(file, step)
#         i += 1
#     return dpoint_i


# def plot_vessels(dpoints):

#     fig_a = plt.figure(figsize=(7, 7))
#     ax_a = fig_a.add_subplot(111, projection='3d')
#     ax_a.grid()

#     for i in range(len(dpoints)):

#         fig = plt.figure(figsize=(7, 7))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.grid()

#         vessel_name, points_vessel = dpoints.get('points{}'.format(i))

#         X, Y, Z = points_vessel[:, 0], points_vessel[:, 1], points_vessel[:, 2]

#         ax_a.plot(X, Y, Z, '--')
#         ax_a.plot(X, Y, Z, 'o', label='control points')

#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')

#         ax.plot(X, Y, Z, '--')
#         ax.plot(X, Y, Z, 'o', label='control points')

#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')

#         plt.title(vessel_name)

#         fig.savefig('N:/vasospasm/pressure_pytec_scripts/plot_paths/plot_' +
#                     vessel_name + '.png')

#         fig_a.savefig(
#             'N:/vasospasm/pressure_pytec_scripts/plot_paths/plot_all.png')


# def division_ICA(pinfo, case):
#     """


#     Parameters
#     ----------
#     pinfo : str, example : 'pt2' , 'vsp7'
#     case : str, 'baseline' or 'vasospasm'

#     Returns
#     -------
#     dpoints_divided : dict of all the control points for the vessels of the patient,
#     with the ICA_MCA --> ICA & MCA for left and right.

#     """

#     if pinfo == 'pt2':
#         folder = '_segmentation_no_vti'
#     else:
#         folder = '_segmentation'
#     pathpath = 'N:/vasospasm/' + pinfo + '/' + case+'/1-geometry/' + \
#         pinfo + '_' + case + folder + '/paths'

#     os.chdir(pathpath)
#     onlyfiles = []
#     for file in glob.glob("*.pth"):
#         onlyfiles.append(file)
#     for files in onlyfiles:
#         if "L_ACA" in files:
#             points_LACA = get_spline_points(files, step)
#         if "R_ACA" in files:
#             points_RACA = get_spline_points(files, step)
#         if "L_ICA_MCA" in files:
#             points_LICAMCA = get_spline_points(files, step)
#         if "R_ICA_MCA" in files:
#             points_RICAMCA = get_spline_points(files, step)

#     ltarget = points_LACA[points_LACA.shape[0] - 1]
#     rtarget = points_RACA[points_RACA.shape[0] - 1]
#     lnorms = []
#     for i in range(points_LICAMCA.shape[0]):
#         lnorm = np.linalg.norm(ltarget - points_LICAMCA[i])
#         lnorms.append(lnorm)
#     lmini = np.min(lnorms)
#     limin = lnorms.index(lmini)
#     points_LICA = points_LICAMCA[:limin]
#     points_LMCA = points_LICAMCA[limin:]
#     rnorms = []
#     for i in range(points_RICAMCA.shape[0]):
#         rnorm = np.linalg.norm(rtarget - points_RICAMCA[i])
#         rnorms.append(rnorm)
#     rmini = np.min(rnorms)
#     imin = rnorms.index(rmini)
#     points_RICA = points_RICAMCA[:imin]
#     points_RMCA = points_RICAMCA[imin:]
#     dpoints_divided = {}
#     k = 0
#     if points_LICA.shape[0] != 0:
#         dpoints_divided["points{}".format(k)] = "L_ICA", points_LICA
#         k += 1
#     if points_RICA.shape[0] != 0:
#         dpoints_divided["points{}".format(k)] = "R_ICA", points_RICA
#         k += 1
#     if points_LMCA.shape[0] != 0:
#         dpoints_divided["points{}".format(k)] = "L_MCA", points_LMCA
#         k += 1
#     if points_RMCA.shape[0] != 0:
#         dpoints_divided["points{}".format(k)] = "R_MCA", points_RMCA
#         k += 1

#     return dpoints_divided


# def get_method_div_A(pinfo, case):
#     """


#     Parameters
#     ----------
#     pinfo : str, example : 'pt2' , 'vsp7'
#     case : str, 'baseline' or 'vasospasm'

#     Returns
#     -------
#     str : 'auto' if there is an Acom, 'manual' if not

#     """
#     dpoints = create_dpoint(pinfo, case)

#     for i in range(len(dpoints)):
#         if "Acom" in dpoints.get("points{}".format(i))[0]:
#             return "auto"
#     return "manual"


# def division_P_2(pinfo, case, vessel):
#     """


#     Parameters
#     ----------
#     pinfo : str. example : 'pt2'
#     case : str. example : 'baseline'
#     vessel : .pth file of the vessel.

#     Returns
#     -------
#     dpoints_divided : dictionary of the control points

#     """

#     dpoints_divided = {}

#     if pinfo == 'pt2':
#         folder = '_segmentation_no_vti'
#     else:
#         folder = '_segmentation'
#     pathpath = 'N:/vasospasm/' + pinfo + '/' + case+'/1-geometry/' + \
#         pinfo + '_' + case + folder + '/paths'

#     target_vessel = vessel[:2] + "PCA"
#     os.chdir(pathpath)
#     onlyfiles = []
#     for file in glob.glob("*.pth"):
#         onlyfiles.append(file)
#     for files in onlyfiles:
#         if vessel in files:
#             points_Pcom = get_spline_points(files, step)

#         if target_vessel in files:
#             points_PCA = get_spline_points(files, step)

#     target = points_Pcom[points_Pcom.shape[0] - 1]
#     norms = []
#     for i in range(points_PCA.shape[0]):
#         norm_i = np.linalg.norm(target - points_PCA[i])
#         norms.append(norm_i)
#     mini = np.min(norms)
#     imin = norms.index(mini)
#     points_P1 = points_PCA[:imin]
#     points_P2 = points_PCA[imin:]
#     dpoints_divided = {}
#     k = 0
#     if points_P1.shape[0] != 0:
#         dpoints_divided["points{}".format(k)] = vessel[:3] + "1", points_P1
#         k += 1
#     if points_P2.shape[0] != 0:
#         dpoints_divided["points{}".format(k)] = vessel[:3] + "2", points_P2
#         k += 1

#     # En l'occurence, le patient n'a pas de rpcom, mais OK parce que RP1 est
#     # vide.

#     return dpoints_divided


def manual_division(pinfo, case, vessel):
    """


    Parameters
    ----------
    pinfo : str. example : 'pt2'
    case : str. example : 'baseline'
    vessel : .pth file of the vessel.

    Returns
    -------
    dpoints_divided :dictionary of the control points
    target :coordinates of the input point

    """
    dpoints_divided = {}

    # pathpath = "C:/Users/Francois/Desktop/Stage_UW/" + pinfo + "/path"

    if pinfo == "pt2":
        folder = "_segmentation_no_vti"
    else:
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
    for files in onlyfiles:
        if vessel in files:
            points_vessel = get_spline_points(files, step)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    X, Y, Z = points_vessel[:, 0], points_vessel[:, 1], points_vessel[:, 2]
    ax.plot(X, Y, Z, "--")
    ax.plot(X, Y, Z, "o")
    plt.title(vessel)
    i_point = 0
    for x, y, z in zip(X, Y, Z):
        annotation = "point {}".format(i_point)
        ax.text(x, y, z, annotation)
        i_point += 1
    plt.show()

    print("\n")
    print("## Select separation point ##   " + vessel[:-4] + "\n")
    for i in range(len(points_vessel)):
        print("   ", i, " : point ", i)

    target = int(input("-->  "))

    points_1 = points_vessel[:target]
    points_2 = points_vessel[target:]
    dpoints_divided = {}
    k = 0
    if points_1.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = vessel[:3] + "1", points_1
        k += 1
    if points_2.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = vessel[:3] + "2", points_2
        k += 1

    return dpoints_divided


# def get_method_div_P(pinfo, case):
#     """


#     Parameters
#     ----------
#     pinfo : str, example : 'pt2' , 'vsp7'
#     case : str, 'baseline' or 'vasospasm'

#     Returns
#     -------
#     L : list of two str, being 'auto' if there is a left/right Pcom, 'manual' if not

#     """

#     dpoints = create_dpoint(pinfo, case)
#     L = ["manual", "manual"]
#     for i in range(len(dpoints)):
#         if "L_Pcom" in dpoints.get("points{}".format(i))[0]:
#             L[0] = "auto"
#         if "R_Pcom" in dpoints.get("points{}".format(i))[0]:
#             L[1] = "auto"
#     return L


# def division_A(pinfo, case):
#     """


#     Parameters
#     ----------
#     pinfo : str, example : 'pt2' , 'vsp7'
#     case : str, 'baseline' or 'vasospasm'

#     Returns
#     -------
#     dpoints_divided : dict of the control points for every vessel

#     """

#     dpoints_divided = {}

#     #pathpath = "C:/Users/Francois/Desktop/Stage_UW/" + pinfo + "/path"

#     if pinfo == 'pt2':
#         folder = '_segmentation_no_vti'
#     else:
#         folder = '_segmentation'
#     pathpath = 'N:/vasospasm/' + pinfo + '/' + case+'/1-geometry/' + \
#         pinfo + '_' + case + folder + '/paths'

#     os.chdir(pathpath)
#     onlyfiles = []
#     for file in glob.glob("*.pth"):
#         onlyfiles.append(file)
#     for files in onlyfiles:
#         if "Acom" in files:
#             points_Acom = get_spline_points(files, step)
#         if "L_ACA" in files:
#             points_LACA = get_spline_points(files, step)
#         if "R_ACA" in files:
#             points_RACA = get_spline_points(files, step)

#     ltarget = points_Acom[points_Acom.shape[0] - 1]
#     rtarget = points_Acom[0]
#     lnorms = []
#     for i in range(points_LACA.shape[0]):
#         lnorm = np.linalg.norm(ltarget - points_LACA[i])
#         lnorms.append(lnorm)
#     lmini = np.min(lnorms)
#     limin = lnorms.index(lmini)
#     points_LA1 = points_LACA[:limin]
#     points_LA2 = points_LACA[limin:]
#     rnorms = []
#     for i in range(points_RACA.shape[0]):
#         rnorm = np.linalg.norm(rtarget - points_RACA[i])
#         rnorms.append(rnorm)
#     rmini = np.min(rnorms)
#     imin = rnorms.index(rmini)
#     points_RA1 = points_RACA[:imin]
#     points_RA2 = points_RACA[imin:]
#     dpoints_divided = {}
#     k = 0
#     if points_LA1.shape[0] != 0:
#         dpoints_divided["points{}".format(k)] = "L_A1", points_LA1
#         k += 1
#     if points_LA2.shape[0] != 0:
#         dpoints_divided["points{}".format(k)] = "L_A2", points_LA2
#         k += 1
#     if points_RA1.shape[0] != 0:
#         dpoints_divided["points{}".format(k)] = "R_A1", points_RA1
#         k += 1
#     if points_RA2.shape[0] != 0:
#         dpoints_divided["points{}".format(k)] = "R_A2", points_RA2
#         k += 1

#     return dpoints_divided


# def division_P_bas(pinfo, case):
#     """


#     Parameters
#     ----------
#     pinfo : str, example : 'pt2' , 'vsp7'
#     case : str, 'baseline' or 'vasospasm'

#     Returns
#     -------
#     dpoints_divided : dict of the control points for every vessel

#     """

#     dpoints_divided = {}

#     #pathpath = "C:/Users/Francois/Desktop/Stage_UW/" + pinfo + "/path"
#     pathpath = 'N:/vasospasm/' + pinfo + '/' + case+ \
#         '/1-geometry/' + pinfo + '_baseline_segmentation_no_vti/paths'
#     os.chdir(pathpath)
#     onlyfiles = []
#     for file in glob.glob("*.pth"):
#         onlyfiles.append(file)
#     for files in onlyfiles:

#         # If one of the PCA is merged with the basilar : separation

#         if ("BAS_L_PCA" or "BAS_R_PCA") in files:
#             points_bas_pca = get_spline_points(files, step)
#             side_bas = files[4:5]
#             for subfile in onlyfiles:

#                 if side_bas == 'L':
#                     if "R_Pcom_PCA" in subfile:
#                         points_target = get_spline_points(subfile, step)

#                 else:
#                     if "L_PCA" in subfile:
#                         points_target = get_spline_points(subfile, step)

#             target = points_target[0]
#             lnorms = []
#             for i in range(points_bas_pca.shape[0]):
#                 lnorm = np.linalg.norm(target - points_bas_pca[i])
#                 lnorms.append(lnorm)
#             lmini = np.min(lnorms)
#             limin = lnorms.index(lmini)
#             points_aca = points_bas_pca[:limin]
#             points_bas = points_bas_pca[limin:]
#             print(points_bas)
#             print('\n')
#             print(points_aca)

#             if "Acom" in files:
#                 points_Acom = get_spline_points(files, step)

#                 if side_bas == 'L':
#                     points_LACA = points_aca
#                     for files in onlyfiles:
#                         if "R_PCA" in files:
#                             points_RACA = get_spline_points(files, step)

#                 else:
#                     points_RACA = points_aca
#                     for files in onlyfiles:
#                         if "L_PCA" in files:
#                             points_LACA = get_spline_points(files, step)

#             ltarget = points_Acom[points_Acom.shape[0] - 1]
#             rtarget = points_Acom[0]
#             lnorms = []
#             for i in range(points_LACA.shape[0]):
#                 lnorm = np.linalg.norm(ltarget - points_LACA[i])
#                 lnorms.append(lnorm)
#             lmini = np.min(lnorms)
#             limin = lnorms.index(lmini)
#             points_LA1 = points_LACA[:limin]
#             points_LA2 = points_LACA[limin:]
#             rnorms = []
#             for i in range(points_RACA.shape[0]):
#                 rnorm = np.linalg.norm(rtarget - points_RACA[i])
#                 rnorms.append(rnorm)
#             rmini = np.min(rnorms)
#             imin = rnorms.index(rmini)
#             points_RA1 = points_RACA[:imin]
#             points_RA2 = points_RACA[imin:]

#             dpoints_divided = {}
#             k = 0
#             if points_bas.shape[0] != 0:
#                 dpoints_divided["points{}".format(
#                     k)] = side_bas + "_BAS", points_bas
#                 k += 1
#             if points_LA1.shape[0] != 0:
#                 dpoints_divided["points{}".format(k)] = "L_P1", points_LA1
#                 k += 1
#             # if points_LA2.shape[0] != 0:
#             #     dpoints_divided["points{}".format(k)] = "L_P2", points_LA2
#             #     k += 1
#             if points_RA1.shape[0] != 0:
#                 dpoints_divided["points{}".format(k)] = "R_P1", points_RA1
#                 k += 1
#             if points_RA2.shape[0] != 0:
#                 dpoints_divided["points{}".format(k)] = "R_P2", points_RA2
#                 k += 1

#             return dpoints_divided

#         else:

#             for subfile in onlyfiles:
#                 if "Acom" in subfile:
#                     points_Acom = get_spline_points(subfile, step)
#                 if "L_PCA" in subfile:
#                     points_LACA = get_spline_points(subfile, step)
#                 if "R_PCA" in subfile:
#                     points_RACA = get_spline_points(subfile, step)

#             ltarget = points_Acom[points_Acom.shape[0] - 1]
#             rtarget = points_Acom[0]
#             lnorms = []
#             for i in range(points_LACA.shape[0]):
#                 lnorm = np.linalg.norm(ltarget - points_LACA[i])
#                 lnorms.append(lnorm)
#             lmini = np.min(lnorms)
#             limin = lnorms.index(lmini)
#             points_LA1 = points_LACA[:limin]
#             points_LA2 = points_LACA[limin:]
#             rnorms = []
#             for i in range(points_RACA.shape[0]):
#                 rnorm = np.linalg.norm(rtarget - points_RACA[i])
#                 rnorms.append(rnorm)
#             rmini = np.min(rnorms)
#             imin = rnorms.index(rmini)
#             points_RA1 = points_RACA[:imin]
#             points_RA2 = points_RACA[imin:]
#             dpoints_divided = {}
#             k = 0
#             if points_LA1.shape[0] != 0:
#                 dpoints_divided["points{}".format(k)] = "L_P1", points_LA1
#                 k += 1
#             if points_LA2.shape[0] != 0:
#                 dpoints_divided["points{}".format(k)] = "L_P2", points_LA2
#                 k += 1
#             if points_RA1.shape[0] != 0:
#                 dpoints_divided["points{}".format(k)] = "R_P1", points_RA1
#                 k += 1
#             if points_RA2.shape[0] != 0:
#                 dpoints_divided["points{}".format(k)] = "R_P2", points_RA2
#                 k += 1

#     return dpoints_divided


# def add_divided_arteries(dpoint_i, dpoints_div):
#     """
#     Parameters
#     ----------
#     dpoint_i : dict of the control points, ICA_MCA release.
#     dpoints_div : dict of the ICA & MCA control points.

#     Returns
#     -------
#     dpoint_i : dict of the control points, ICA_MCA & ICA & MCA release.

#     """

#     counterbalance = 0
#     for j in range(len(dpoint_i), len(dpoint_i) + len(dpoints_div)):
#         k = j - len(dpoint_i) + counterbalance
#         dpoint_i["points{}".format(j)] = (
#             dpoints_div.get("points{}".format(k))[0],
#             dpoints_div.get("points{}".format(k))[1],
#         )
#         counterbalance += 1
#     return dpoint_i


# def delete_old_arteries(dpoint_i):
#     """
#     Parameters
#     ----------
#     dpoint_i : dict of the control point, ICA_MCA & ICA & MCA release.

#     Returns
#     -------
#     dpoint_i : dict of the control points, ICA & MCA without ICA_MCA release.
#     indices : list of the indices which are kept after deleting the fusionned arteries.

#     """

#     I_supp = []
#     for j in range(len(dpoint_i)):

#         if "ICA_MCA" in dpoint_i.get("points{}".format(j))[0]:

#             # del dpoint_i["points{}".format(j)]
#             I_supp.append(j)

#         if (
#             dpoint_i.get("points{}".format(j))[0] is not None
#             and "L_ACA" in dpoint_i.get("points{}".format(j))[0]
#         ):

#             # del dpoint_i["points{}".format(j)]
#             I_supp.append(j)

#         if (
#             dpoint_i.get("points{}".format(j))[0] is not None
#             and "R_ACA" in dpoint_i.get("points{}".format(j))[0]
#         ):

#             # del dpoint_i["points{}".format(j)]
#             I_supp.append(j)

#         if (
#             dpoint_i.get("points{}".format(j))[0] is not None
#             and "L_PCA" in dpoint_i.get("points{}".format(j))[0]
#         ):

#             # del dpoint_i["points{}".format(j)]
#             I_supp.append(j)

#         if (
#             dpoint_i.get("points{}".format(j))[0] is not None
#             and "R_PCA" in dpoint_i.get("points{}".format(j))[0]
#         ):
#             I_supp.append(j)

#         if (
#             dpoint_i.get("points{}".format(j))[0] is not None
#             and "aneurysm" in dpoint_i.get("points{}".format(j))[0]
#         ):
#             I_supp.append(j)

#     for i in I_supp:
#         del dpoint_i["points{}".format(i)]
#     indices = [i for i in range(
#         len(dpoint_i) + len(I_supp)) if i not in I_supp]
#     return dpoint_i, indices


# def createfinal_dicts(dpoint_i, indices):
#     """

#     Parameters
#     ----------
#     dpoint_i : dict of the control point, ICA & MCA without ICA_MCA release.
#     indices :  list of the indices which are kept after deleting the fusionned arteries.


#     Returns
#     -------
#     dpoints : dict of the control points  ICA & MCA without ICA_MCA release, with the accurate indexation .
#     dvectors : dict of the normal vectors  ICA & MCA without ICA_MCA release, with the accurate indexation .


#     """

#     j = 0
#     dpoints = {}
#     dvectors = {}
#     for i in indices:
#         filename, points = (
#             dpoint_i.get("points{}".format(i))[0],
#             dpoint_i.get("points{}".format(i))[1],
#         )
#         dpoints["points{}".format(j)] = filename, points
#         dvectors["vectors{}".format(
#             j)] = filename, calculate_normal_vectors(points)
#         j += 1
#     return dpoints, dvectors


def get_list_files_dat(pinfo, case, num_cycle):
    """


    Parameters
    ----------
    pinfo : str, patient information, composed of pt/vsp + number.
    num_cycle : int, number of the cycle computed
    case : str, baseline or vasospasm

    Returns
    -------
    onlyfiles : list of .dat files for the current patient, for the case and cycle wanted.
    """

    num_cycle = str(num_cycle)
    case = "baseline"
    path = "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit/"
    os.chdir(path)
    onlyfiles = []

    for file in glob.glob("*.dat"):
        if pinfo + "_" + case + "_cycle" + num_cycle in file:
            onlyfiles.append(file)
    indices = [l[13:-4] for l in onlyfiles]

    for i in range(len(indices)):
        newpath = (
            "N:/vasospasm/pressure_pytec_scripts/plots_c/"
            + pinfo
            + "/"
            + case
            + "/cycle_"
            + num_cycle
            + "/plot_"
            + indices[i]
        )
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    return onlyfiles, indices


def save_dict(dico, name):
    """


    Parameters
    ----------
    dico : dictionary one wants to save
    name : str. path + name of the dictionary

    Returns
    -------
    None.

    """

    with open(name + ".pkl", "wb") as f:
        pickle.dump(dico, f)


def load_dict(name):
    """


    Parameters
    ----------
    name : str. path + name of the dictionary one wants to load

    Returns
    -------
    b : the loaded dictionary

    """
    with open(name + ".pkl", "rb") as handle:
        b = pickle.load(handle)
    return b


def get_variation(pinfo, case):

    dinfo = scipy.io.loadmat(
        "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/case_info.mat"
    )
    variation = int(dinfo.get("variation_input"))

    return variation


# %% Functions Tecplot


def change_name(name, Lfiles):

    x = name.lower()
    x = x[0] + x[2:]
    # L=[z[13:] for z in Lfiles[1:]]

    if "lsup_cer" in x:
        name_f = "lsc"
    elif "lsupcer" in x:
        name_f = "lsc"
    elif "rsup_cer" in x:
        name_f = "rsc"
    elif "rsupcer" in x:
        name_f = "rsc"
    elif "rsup_cereb_duplicate" in x:
        name_f = "rsc2"
    elif "rp2" in x:
        name_f = "rpca"

    elif "ra2" in x:
        name_f = "raca"

    elif "lp2" in x:
        name_f = "lpca"

    elif "la2" in x:
        name_f = "lpca"

    elif "rop" in x:
        name_f = "rop"

    elif "lop" in x:
        name_f = "lop"

    else:
        name_f = x

    iL = 0
    for y in Lfiles:
        if name_f in y:
            iL = Lfiles.index(y)
            break
    if iL == 0:
        iL = Lfiles.index(pinfo + "_" + case + ".walls")

    return Lfiles[iL]


def plot_other(data_file, vessel_name):
    zones = data_file.zone_names

    ZERO = np.zeros((3, 1))

    name = change_name(vessel_name, zones)

    name_walls = pinfo + "_" + case + ".walls"
    if name != name_walls:

        cx = data_file.zone(name).values("X")[:]
        cy = data_file.zone(name).values("Y")[:]
        cz = data_file.zone(name).values("Z")[:]
        x_base = np.asarray(cx)
        y_base = np.asarray(cy)
        z_base = np.asarray(cz)

        coordinates_fluent = np.array([x_base, y_base, z_base]).T

        # fig=plt.figure(figsize=(7,7))
        # ax=fig.add_subplot(111,projection='3d')
        # ax.grid()
        # ax.scatter(x_base,y_base,z_base)
        # ax.plot_trisurf(x_base,y_base,z_base)

        xm = np.mean(x_base)
        ym = np.mean(y_base)
        zm = np.mean(z_base)
        center = np.array([xm, ym, zm])

        # ax.plot(xm,ym,zm,'x')
        L = []
        for i in range(coordinates_fluent.shape[0]):
            b = np.linalg.norm(coordinates_fluent[i, :] - center)
            L.append(b)

        lmin = np.min(L)
        # print('norme mini : ',lmin)
        imin = L.index(lmin)

        xyz_frontier = np.array(
            [
                coordinates_fluent[imin, 0],
                coordinates_fluent[imin, 1],
                coordinates_fluent[imin, 2],
            ]
        )
        # ax.plot(coordinates_fluent[imin,0],coordinates_fluent[imin,1],coordinates_fluent[imin,2],'o')

        # plt.title(name)
        # plt.show()

        return xyz_frontier

    return ZERO


def get_starting_indice(i_vessel, dpoints_i, dvectors_i, data_file_baseline):

    array_points = dpoints_i.get("points{}".format(i_vessel))
    array_vectors = dvectors_i.get("vectors{}".format(i_vessel))
    name = dpoints_i.get("points{}".format(i_vessel))[0]
    xyz_frontier = plot_other(data_file_baseline, name)

    n_ = array_points[1].shape[0]

    L = []
    for i in range(array_points[1].shape[0]):
        b = np.linalg.norm(array_points[1][i, :] - xyz_frontier)
        L.append(b)

    lmin = np.min(L)
    starting_indice = L.index(lmin)

    # fig=plt.figure(figsize=(7,7))
    # ax=fig.add_subplot(111,projection='3d')
    # ax.grid()
    # ax.scatter(array_points[1][:,0],array_points[1][:,1],array_points[1][:,2])
    # ax.scatter(xyz_frontier[0],xyz_frontier[1],xyz_frontier[2])

    # plt.title(name)
    # plt.show()

    return starting_indice, lmin


def get_origin(dpoints_i, dvectors_i, i_vessel, data_file_baseline):

    starting_indice, lmin = get_starting_indice(
        i_vessel, dpoints_i, dvectors_i, data_file_baseline
    )

    array_points = dpoints_i.get("points{}".format(i_vessel))

    norms = calculate_norms(dvectors_i.get("vectors{}".format(i_vessel))[1])
    avg_norm = np.mean(norms)
    name = dpoints_i.get("points{}".format(i_vessel))[0]
    ZERO = np.zeros((3, 1))
    ratio = avg_norm / lmin
    if (
        not np.array_equal(plot_other(data_file_baseline, name), ZERO)
        and ratio > 0.05 * step
    ):

        n_ = array_points[1].shape[0]

        starting_indice, lmin = get_starting_indice(
            i_vessel, dpoints_i, dvectors_i, data_file_baseline
        )

        # if starting_indice
        # A faire : definir si l'artere est concernee par la mise a jour par outlet ok
        #           Definir si on a un starting ou ending point ok
        # definir les nouveaux dpoints et dvectors puis reprendre la structure de base
        # Esaayer de load depuis une fonction

        coef = abs(starting_indice - n_) / n_

        if coef >= 0.5:

            # shape of n - indice of start + 1 to insert the boundary  point
            origin_up = np.zeros((n_ - starting_indice, 3))
            # origin_up[0,:]=xyz_frontier
            origin_up[:, :] = array_points[1][starting_indice:, :]
        else:
            # If the file is read in the other way : the starting point is a
            # ending point
            ending_indice = starting_indice
            # The shape of the array of points is now ending + 1
            origin_up = np.zeros((ending_indice, 3))
            len_origin = origin_up.shape[0]
            # origin_up[len_origin-1,:]=xyz_frontier
            origin_up[:len_origin, :] = array_points[1][:ending_indice, :]

        return origin_up

    return array_points[1]


def data_coor(data_file):
    """


    Parameters
    ----------
    data_file : data_file created when loading data on tecplot

    Returns
    -------
    coordinates_fluent : coordinates of all the points or the fluent simulation

    """

    name = pinfo + "_" + case + ".walls"

    # print('chosen file : ', name)

    cx = data_file.zone(name).values("X")[:]
    cy = data_file.zone(name).values("Y")[:]
    cz = data_file.zone(name).values("Z")[:]
    x_base = np.asarray(cx)
    y_base = np.asarray(cy)
    z_base = np.asarray(cz)

    coordinates_fluent = np.array([x_base, y_base, z_base]).T

    return coordinates_fluent


def find_closest(origin, name):
    """


    Parameters
    ----------
    origin :coordinates of the control point on which one want to work

    Returns
    -------
    coordinates of the closest point to the control points, by minimazation with the euclidean distance

    """

    L = []
    coordinates_fluent = data_coor(data_file)
    for i in range(coordinates_fluent.shape[0]):
        b = np.linalg.norm(coordinates_fluent[i, :] - origin)
        L.append(b)

    lmin = np.min(L)
    imin = L.index(lmin)

    return coordinates_fluent[imin, :]


def find_slice(slices, origin):
    """


    Parameters
    ----------
    slices : type : generator, slice generator.
    origin : coordinates of the origin of the slice searched.

    Returns
    -------
    dict_slice : dictionnary of the slices form the generator.
    imin : index of the searched slice.

    """
    L = []
    dict_slice = {}
    i = 0
    for s in slices:
        dict_slice["{}".format(i)] = s
        (x, y, z) = (
            np.mean(s.values("X")[:]),
            np.mean(s.values("Y")[:]),
            np.mean(s.values("Z")[:]),
        )
        L.append(np.array([x, y, z]))
        i += 1

    Lnorm = [np.linalg.norm(x - origin) for x in L]
    mini = min(Lnorm)
    imin = Lnorm.index(mini)

    return dict_slice, imin


def get_pressure(origin, normal, name):
    """
    Compute the pressure in a given slice.
    Assuming that every data that could be needed from fluent are loaded.


    Parameters
    ----------
    origin : (3,1) array of the origin coordinates of the slice.
    vectors : (3,1) array of the normal vector coordinate

    Returns
    -------
    prssure : average pressure value in the slice
    """

    frame = tp.active_frame()
    plot = frame.plot()

    plot.show_slices = True
    slices = plot.slices(0)
    slices.show = True

    origin_s = find_closest(origin, name)
    origin_slice = (origin_s[0], origin_s[1], origin_s[2])
    normal = (normal[0], normal[1], normal[2])

    slices_0 = tp.data.extract.extract_slice(
        mode=ExtractMode.OneZonePerConnectedRegion,
        origin=origin_slice,
        normal=normal,
        dataset=data_file,
    )

    dict_slice, n_iteration = find_slice(slices_0, origin_slice)

    final_slice = dict_slice.get("{}".format(n_iteration))
    min_pressure = np.min(final_slice.values("Pressure")[:])
    avg_pressure = np.mean(final_slice.values("Pressure")[:])
    max_pressure = np.max(final_slice.values("Pressure")[:])

    # Extract the 3D array of the points of the slice (test)

    if "ICA" in name:

        x_arr = final_slice.values("x")[:]
        y_arr = final_slice.values("y")[:]
        z_arr = final_slice.values("z")[:]

        array_slice = np.array((x_arr, y_arr, z_arr))

    tp.macro.execute_command("$!RedrawAll")
    # Attention mettre à jour le return
    return min_pressure, avg_pressure, max_pressure, array_slice


def compute_along(i_vessel, dpoints, dvectors):
    """


    Parameters
    ----------
    i_vessel :index of the vessel.

    Returns
    -------
    Lpressure : list of the average pressure along the vessel.

    """

    name = dpoints.get("points{}".format(i_vessel))[0]

    # if 'ICA' in name:

    # x_arr=final_slice.values('x')[:]
    # y_arr=final_slice.values('y')[:]
    # z_arr=final_slice.values('z')[:]

    # array_slice=np.array((x_arr,y_arr,z_arr))

    # center=geom.find_center(array_slice)
    # radius=geom.find_radius(center,array_slice)

    Lavg, Lmin, Lmax = [], [], []

    # starting_indice=get_starting_indice(i_vessel)

    # print('starting_indice : ',starting_indice)

    # Nouveaux points
    # points = get_origin(i_vessel)
    # vectors=calculate_normal_vectors(points)

    points = dpoints.get("points{}".format(i_vessel))[1]
    vectors = dvectors.get("vectors{}".format(i_vessel))[1]
    dslice = {}
    n_ = points.shape[0]
    if n_ > 2:
        print(
            "### Compute on ",
            dpoints.get("points{}".format(i_vessel))[0],
            " ### Vessel ",
            i_vessel,
            "/",
            len(dpoints),
            "\n",
        )

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.grid()

        for j in tqdm(range(len(points) - 1)):
            # If ICA -> take longer normal vectors to avoid the anomaly
            # if 'ICA' in name:
            #     if j > 1 and j<len(points)-2 :
            #         origin = points[j, :]
            #         normal = (vectors[j-1, :]+vectors[j+1,:])*0.5
            #     else:
            #         origin = points[j, :]
            #         normal = vectors[j, :]
            # else:
            origin = points[j, :]
            normal = vectors[j, :]
            min_pressure, avg_pressure, max_pressure, array_slice = get_pressure(
                origin, normal, name
            )
            print("   $$ Control point ", j, " : Pressure = ", avg_pressure)
            Lmin.append(min_pressure)
            Lavg.append(avg_pressure)
            Lmax.append(max_pressure)

            Lpressure = np.array([Lmin, Lavg, Lmax])

            ax.scatter(
                array_slice[0, :],
                array_slice[1, :],
                array_slice[2, :],
                c="b",
                linewidths=0.5,
            )
            ax.scatter(origin[0], origin[1], origin[2], c="k")

            dslice["point{}".format(j)] = array_slice
            ax.view_init(30, 30)
        ax.legend()
        plt.show()
        print("\n")

    else:
        L = [0] * (len(points))
        Lpressure = np.array([L, L, L])

    print(Lpressure)

    return Lpressure, dslice


def plot_linear(i_vessel, Lpress, dpoints_u, dvectors_u, dist):
    """


    Parameters
    ----------
    i_vessel : index of the vessel.
    Lpress : array of the pressures along the vessel.

    Returns
    -------
    fig : plot of the pressure along the vessel, linear interpolation

    """

    Ldist = []
    # starting_indice,xyz_frontier = get_starting_indice(i_vessel,dpoints,dvectors)

    # origin=get_origin(i_vessel)
    # vectors=calculate_normal_vectors(origin)

    # remplacer par dpoints et dvectors updated

    origin = dpoints_u.get("points{}".format(i_vessel))[1]
    vectors = dvectors_u.get("vectors{}".format(i_vessel))[1]

    for i in range(0, dist.shape[0] - 1):
        Ldist.append(float((dist[i] + dist[i + 1]) / 2))

    fig = plt.figure(figsize=(14.4, 10.8))

    # plt.plot(Ldist, Lpress[0,:], '--',label='Minimum')
    # plt.plot(Ldist, Lpress[0,:], '^', label='Minimum')

    plt.plot(Ldist, Lpress[1, :], "--")
    plt.plot(Ldist, Lpress[1, :], "o", label="Average")

    # plt.plot(Ldist, Lpress[2,:], '--',label='Maximum')
    # plt.plot(Ldist, Lpress[2,:], 'v', label='Maximum')

    plt.fill_between(Ldist, Lpress[0, :], Lpress[2, :], alpha=0.2)

    plt.grid()
    plt.xlabel("distance along the vessel (m)", fontsize=18)
    plt.ylabel("Pressure", fontsize=18)
    plt.title(
        "Pressure along the " + ddist.get("dist{}".format(i_vessel))[0], fontsize=20
    )
    plt.legend(loc="best")

    plt.show()

    return fig


def save_pressure(i, dpoints, dvectors):
    """


    Parameters
    ----------
    i : vessel index.

    Returns
    -------
    dpressure : dictionnary of the pressure in the vessel i.

    """
    dpressure = {}
    # dpressure['Informations']=pinfo,case,filename
    Lpress, dslice = compute_along(i, dpoints, dvectors)

    pressure_array = invert_array(np.array(Lpress))
    name = dpoints.get("points{}".format(i))[0]
    dpressure["pressure{}".format(i)] = name, pressure_array

    return dpressure, dslice


def save_pressure_all(dpoints, dvectors):
    """


    Returns
    -------
    dpressure : dictionary of the pressure in all the vessels

    """
    dpressure = {}
    for i in range(len(dpoints)):
        pressure_array = invert_array(np.array(compute_along(i, dpoints, dvectors)[0]))
        name = dpoints.get("points{}".format(i))[0]
        dpressure["pressure{}".format(i)] = name, pressure_array

    return dpressure


def plot_on_time(dpressure, i_vessel, pinfo):

    onlydat, indices = get_list_files_dat(pinfo, case, num_cycle)
    len_vessel = (
        dpressure.get("{}".format(indices[0]))
        .get("pressure{}".format(i_vessel))[1]
        .shape[0]
    )
    name_vessel = dpressure.get("{}".format(indices[0])).get(
        "pressure{}".format(i_vessel)
    )[0]
    tabY = np.zeros((len_vessel))
    X = np.linspace(1, len(onlydat) - 1, len(onlydat) - 1)

    for i in range(len_vessel):

        Y = [
            dpressure.get("{}".format(indices[j])).get("pressure{}".format(i_vessel))[
                1
            ][i, 1]
            for j in range(len(onlydat) - 1)
        ]
        # modify by taking the second line (avg)
        plt.plot(X, Y)
    plt.grid()

    plt.xlabel("Time step")
    plt.ylabel("Pressure")
    plt.title("Pressure in each point in function of time in the " + name_vessel)
    plt.savefig(
        "N:/vasospasm/pressure_pytec_scripts/plots_time/"
        + pinfo
        + "_"
        + case
        + "_"
        + name_vessel
        + ".png"
    )
    plt.show()


def plot_time_dispersion(dpressure, i_vessel, pinfo):

    Ldist = []

    onlydat, indices = get_list_files_dat(pinfo, case, num_cycle)
    len_vessel = (
        dpressure.get("{}".format(indices[0]))
        .get("pressure{}".format(i_vessel))[1]
        .shape[1]
    )
    name_vessel = dpressure.get("{}".format(indices[0])).get(
        "pressure{}".format(i_vessel)
    )[0]

    dist = ddist.get("dist{}".format(i_vessel))[1]
    for i in range(0, dist.shape[0] - 1):
        Ldist.append(float((dist[i] + dist[i + 1]) / 2))

    tab_pressure = np.zeros((len_vessel, 3))

    for i in range(len_vessel):
        Lmin = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][0, i]
            for k in range(len(onlydat) - 1)
        ]
        Lmean = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1, i]
            for k in range(len(onlydat) - 1)
        ]
        Lmax = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][2, i]
            for k in range(len(onlydat) - 1)
        ]

        tab_pressure[i, 0] = sum(Lmin) / len(Lmin)
        tab_pressure[i, 1] = sum(Lmean) / len(Lmean)
        tab_pressure[i, 2] = sum(Lmax) / len(Lmax)

    fig = plt.figure(figsize=(14.4, 10.8))

    plt.plot(Ldist, tab_pressure[:, 1], "--")
    plt.plot(Ldist, tab_pressure[:, 1], "o", label="Average pressure over time")
    plt.fill_between(
        Ldist,
        tab_pressure[:, 0],
        tab_pressure[:, 2],
        alpha=0.2,
        label="average enveloppe of min/max pressure over time",
    )

    plt.grid()
    plt.xlabel("distance along the vessel (m)", fontsize=18)
    plt.ylabel("Pressure", fontsize=18)
    plt.title(
        "Pressure along the " + ddist.get("dist{}".format(i_vessel))[0], fontsize=20
    )
    plt.legend(loc="best")

    plt.savefig(
        "N:/vasospasm/pressure_pytec_scripts/plots_avg/"
        + pinfo
        + "_"
        + case
        + "_"
        + name_vessel
        + ".png"
    )
    plt.show()

    return fig


def invert_array(arr):
    new_arr = np.ones_like(arr)
    if arr[1, 0] < arr[1, arr.shape[1] - 1]:
        for i in range(3):
            new_arr[i, :] = arr[i][::-1]
        return 1, new_arr
    else:
        return 0, arr


def get_Q_final(pinfo, case):

    dinfo = scipy.io.loadmat(
        "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/case_info.mat"
    )
    dqfinal = dinfo.get("Q_final")
    Q_arr = np.zeros((30, 11))

    for i in range(0, dqfinal.shape[0] // 2 - 3, 3):
        p = i // 3
        Q_arr[p][:] = dqfinal[i][:]

    for i in range(dqfinal.shape[0] // 2 + 9, dqfinal.shape[0], 3):
        p = i // 3 - 3
        Q_arr[p][:] = dqfinal[i][:]

    return Q_arr


def plot_R(dpressure, i_vessel, pinfo, case):

    Ldist = []

    onlydat, indices = get_list_files_dat(pinfo, case, num_cycle)
    len_vessel = (
        dpressure.get("{}".format(indices[0]))
        .get("pressure{}".format(i_vessel))[1][1]
        .shape[1]
    )
    name_vessel = dpressure.get("{}".format(indices[0])).get(
        "pressure{}".format(i_vessel)
    )[0]

    dist = ddist.get("dist{}".format(i_vessel))[1]
    for i in range(0, dist.shape[0] - 1):
        Ldist.append(float((dist[i] + dist[i + 1]) / 2))

    tab_pressure = np.zeros((len_vessel, 3))

    Qfinal = np.mean(get_Q_final(pinfo, case)[:, 3])
    len_cycle = 30
    for i in range(len_vessel):
        # print(dpressure.get('{}'.format(indices[0])).get('pressure{}'.format(i_vessel))[1][1][0,i])
        Lmin = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][0, i]
            for k in range(len_cycle)
        ]
        Lmean = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][1, i]
            for k in range(len_cycle)
        ]
        Lmax = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][2, i]
            for k in range(len_cycle)
        ]

        tab_pressure[i, 0] = sum(Lmin) / len(Lmin)
        tab_pressure[i, 1] = sum(Lmean) / len(Lmean)
        tab_pressure[i, 2] = sum(Lmax) / len(Lmax)

    tab_resistance = np.zeros((len_vessel, 3))
    tab_resist_locale = np.zeros((len_vessel - 1, 3))
    for i in range(1, len_vessel):
        tab_resistance[i, :] = (tab_pressure[i, :] - tab_pressure[1, :]) / Qfinal
    for i in range(len_vessel - 1):
        tab_resist_locale[i, :] = (tab_pressure[i + 1, :] - tab_pressure[i, :]) / Qfinal

    fig = plt.figure(figsize=(14.4, 10.8))

    plt.plot(Ldist[:-1], tab_resist_locale[:, 1], label="local resistance")

    plt.plot(Ldist[1:], tab_resistance[1:, 1], "--")
    plt.plot(
        Ldist[1:], tab_resistance[1:, 1], "o", label="Average resistance over time"
    )
    # plt.fill_between(Ldist,
    #                  tab_resistance[:,
    #                                 0],
    #                  tab_resistance[:,
    #                                 2],
    #                  alpha=0.2,
    #                  label='average enveloppe of min/max resistance over time')

    plt.grid()
    plt.xlabel("distance along the vessel (m)", fontsize=18)
    plt.ylabel("Resistance", fontsize=18)
    plt.title("Resistance along the " + name_vessel, fontsize=20)
    plt.legend(loc="best")

    plt.savefig(
        "N:/vasospasm/pressure_pytec_scripts/plots_avg/"
        + pinfo
        + "_"
        + case
        + "_"
        + name_vessel
        + ".png"
    )
    plt.show()

    return fig


# %% Main


def main():
    """

    The object of this script is to compute and plot the pressure along the vessels
    of the circle of Willis of a specific patient for certain points.



    This program
        --> Step 1 : Extract files and coordinates, label names. All in a dictionnary

        --> Step 2: Operate a divison for ICA_MCA --> ICA & MCA, PCA --> P1 & P2 and ACA --> A1 & A2.

    These division are made automatically if there is the right vessel to make the separation :
        - ACA for the ICA_MCA
        - Pcom for the PCA
        - Acom for the ACA
    The separation is made finding the closest point of the unsparated vessel from
    the first/last point of the vessel used to do the separation, by minimazing the euclidian distances.

    If there is a vessel missing, the separation is made manually by the user, which enter the coordinates
    of the separation point.

        --> Step 3 : Add the divided ones in coordinates dictionnary, and remove the ICA_MCA/PCA/ACAs

        --> Step 4 : Compute the vectors dictionnary

    This is just made by substracting the coordinates of the points terms to terms.

        --> Step 5 : Compute pressure with tecplot

            Step 5.1 : Selecting the good case (patient, case, num_cycle, .dat file), and load data into Tecplot
            Step 5.2 : Find the list of the closest points in fluent to the control points.
            Step 5.3 : Make a slice for each control point and normal vector
            Step 5.4 : Find the subslice that correspond to the vessel in which one is interested
            Step 5.5 : Compute the average pressure in the subslice
            Step 5.6 : Save the pressures in a dictionnary,containing every cycle/.dat file/ vessel.
            Step 5.7 : change the order of the vessel if necessary to have a decreasing plot of pressure
        --> Step 6 : Plot pressure in each vessel

            Step 7 : extract flowrate, compute delta_P, and plot time average resistance along the segments.

    """

    global pinfo
    global case
    global num_cycle
    global select_file
    global filename
    global data_file
    global ddist
    global step
    print("$ Patient informations $\n")
    ptype = input("Patient origin? -- v : vsp## or p: pt## --\n")
    if ptype == "v":
        ptype = "vsp"
    elif ptype == "p":
        ptype = "pt"
    pnumber = input("Patient number?\n")
    pinfo = ptype + pnumber
    case = input("Case ? -- b for baseline of v for vasospasm\n")
    if case == "b":
        case = "baseline"
    elif case == "v":
        case = "vasospasm"

    print("$ Select computing cases\n")
    num_cycle = int(input("Which cycle ? 1,2,3 or 4\n"))

    onlydat, indices_dat = get_list_files_dat(pinfo, case, num_cycle)

    for k in range(len(onlydat)):
        print(k, ": " + indices_dat[k][9:-3] + ":" + indices_dat[k][11:] + " s")
    print("a : one period (30 time steps) \n")
    select_file = input("which time step?\n")

    print("$ Step $\n")
    step = int(input())

    # Load a different module of the control points extraction and sorting depending on the patient variation

    variation = get_variation(pinfo, case)

    os.chdir("N:/vasospasm/pressure_pytec_scripts")
    module_name = "division_variation" + str(variation)
    module = importlib.import_module(module_name)

    importlib.reload(module)
    dpoints_u, dvectors_u = module._main_(pinfo, case, step)

    # return dpoints,dvectors

    # Load baseline.dat to compare points with outlet, and update the
    # dictionaries

    # print(
    #     ' ############  Step 1 : First Connection to Tecplot  ############ Initializing data : Final Geometry and distances along the vessels \n')

    # filename = pinfo + '_' + case + '.dat'
    # logging.basicConfig(level=logging.DEBUG)

    # # Run this script with "-c" to connect to Tecplot 360 on port 7600
    # # To enable connections in Tecplot 360, click on:
    # #   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

    # tp.session.connect()
    # tp.new_layout()
    # frame = tp.active_frame()

    # dir_file = 'N:/vasospasm/' + pinfo + '/' + case+ \
    #     '/3-computational/hyak_submit/' + filename

    # data_file_baseline = tp.data.load_fluent(
    #     case_filenames=[
    #         'N:/vasospasm/' +
    #         pinfo +
    #         '/' +
    #         case+
    #         '/3-computational/hyak_submit/' +
    #         pinfo +
    #         '_' +
    #         case+
    #         '.cas'],
    #     data_filenames=[dir_file])

    # dpoints_u = {}
    # dvectors_u = {}
    # dnorms_u = {}

    # for l in tqdm(range(len(dpoints))):
    #     dpoints_u['points{}'.format(l)] = dpoints.get('points{}'.format(l))[
    #         0], get_origin(dpoints, dvectors, l, data_file_baseline)
    #     dvectors_u['vectors{}'.format(l)] = dpoints.get('points{}'.format(l))[
    #         0], calculate_normal_vectors(get_origin(dpoints, dvectors, l, data_file_baseline))
    #     dnorms_u['norms{}'.format(l)] = dpoints.get('points{}'.format(l))[
    #         0], calculate_norms(dvectors_u.get('vectors{}'.format(l))[1])

    # #ddist = get_distance_along(dvectors_u)

    # Lnorm = []
    # for l in tqdm(range(len(dpoints))):
    #     Lnorm.append(np.mean(dnorms_u.get('norms{}'.format(l))[1]))

    # print('\n')

    # definition = sum(Lnorm) / len(Lnorm)

    # print('spatial step (m) : ', definition, '\n')

    for k in range(len(dpoints_u)):
        print(k, " : " + dpoints_u.get("points{}".format(k))[0])
    print("a : all vessels\n")
    select_vessel = input("Compute on which vessel ?\n")

    # Step 5#

    dpressure = {}
    dpressure["Informations"] = pinfo, case

    if select_file == "a":
        start = 0
        end = 30
    else:
        i_file = int(select_file)
        start = i_file
        end = i_file + 1

    for i in range(start, end):

        filename = onlydat[i]

        print(
            " ############  Step 2 : Connection to Tecplot  ############ Time step : ",
            i,
            "\n",
        )
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

        data_file = tp.data.load_fluent(
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

        tp.macro.execute_command("$!RedrawAll")
        tp.active_frame().plot().rgb_coloring.red_variable_index = 3
        tp.active_frame().plot().rgb_coloring.green_variable_index = 3
        tp.active_frame().plot().rgb_coloring.blue_variable_index = 13
        tp.active_frame().plot().contour(0).variable_index = 3
        tp.active_frame().plot().contour(1).variable_index = 4
        tp.active_frame().plot().contour(2).variable_index = 5
        tp.active_frame().plot().contour(3).variable_index = 6
        tp.active_frame().plot().contour(4).variable_index = 7
        tp.active_frame().plot().contour(5).variable_index = 8
        tp.active_frame().plot().contour(6).variable_index = 9
        tp.active_frame().plot().contour(7).variable_index = 10
        tp.active_frame().plot().show_contour = True
        tp.macro.execute_command("$!RedrawAll")
        tp.active_frame().plot(PlotType.Cartesian3D).show_slices = True
        slices = frame.plot().slices()
        slices.contour.show = True
        frame = tp.active_frame()
        plot = frame.plot()
        plot.show_slices = True
        slices = plot.slices(0)
        slices.show = True

        print(" ############  Step 3 : Compute pressure  ############\n")

        if select_vessel == "a":
            dpressure["{}".format(indices_dat[i])] = save_pressure_all(
                dpoints_u, dvectors_u
            )
            dname = "".join(["dpressure", "_", pinfo, "_", case])
            save_dict(dpressure, "N:/vasospasm/pressure_pytec_scripts/" + dname)

        else:
            i_ = int(select_vessel)
            dpress, dslice = save_pressure(i_, dpoints_u, dvectors_u)
            dpressure["{}".format(indices_dat[i])] = dpress
            save_dict(
                dpressure,
                "N:/vasospasm/pressure_pytec_scripts/dpressure" + select_vessel,
            )

        if select_vessel == "a":
            ddist = {}
            for k in range(len(dpoints_u)):
                ddist["{}".format(k)] = get_distance_along(
                    k, i, dpressure, dvectors_u, dpoints_u
                )

        else:
            i_vessel = int(select_vessel)
            ddist = get_distance_along(i_vessel, i, dpressure, dvectors_u, dpoints_u)

        if select_vessel == "a":
            for k in range(len(dpoints_u)):
                Lpress = dpressure.get("{}".format(indices_dat[i])).get(
                    "pressure{}".format(k)
                )[1][1]
                dist = ddist.get("{}".format(k)).get("dist{}".format(k))[1]
                fig = plot_linear(k, Lpress, dpoints_u, dvectors_u, dist)
                fig.savefig(
                    "N:/vasospasm/pressure_pytec_scripts/plots_c/"
                    + pinfo
                    + "/"
                    + case
                    + "/cycle_"
                    + str(num_cycle)
                    + "/plot_"
                    + indices_dat[i]
                    + "/"
                    + "plot_"
                    + dpoints_u.get("points{}".format(k))[0]
                    + ".png"
                )

        else:
            i_vessel = int(select_vessel)
            Lpress = dpressure.get("{}".format(indices_dat[i])).get(
                "pressure{}".format(i_)
            )[1][1]
            dist = ddist.get("dist{}".format(i_vessel))[1]
            print("Lpress : ", Lpress)
            print("dist :", dist)
            fig = plot_linear(i_vessel, Lpress, dpoints_u, dvectors_u, dist)
            fig.savefig(
                "N:/vasospasm/pressure_pytec_scripts/plots_c/"
                + pinfo
                + "/"
                + case
                + "/cycle_"
                + str(num_cycle)
                + "/plot_"
                + indices_dat[i]
                + "/"
                + "plot_"
                + dpoints_u.get("points{}".format(i_))[0]
                + ".png"
            )

    # plot_R(dpressure,i_vessel,pinfo,case)

    return dpressure


#%% Test new method slice ICA

array_init = dpress_init.get("cycle2-1-02257").get("pressure8")[1][1][1]
array_new = dpress_new.get("cycle2-1-02257").get("pressure8")[1][1][1]
Lerr = []
for i in range(array_init.shape[0]):
    err = 100 * abs(array_init[i] - array_new[i]) / array_init[i]
    Lerr.append(err)

X = np.linspace(0, array_init.shape[0] - 1, array_init.shape[0])
plt.plot(X, Lerr)
# plt.plot(X,array_init,label='original')
# plt.plot(X,array_new,label='after method')
plt.title("Space between the two methods")
plt.ylabel("% of error")
# plt.fill_between(X,array_init,array_new)
plt.show()


MSE = 0
for i in range(array_init.shape[0]):
    MSE += array_init[i] - array_new[i]

#%% Test slices

# Plot tecplot slices


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
points = dpoints.get("points8")[1]

X = points[47, 0]
Y = points[47, 1]
Z = points[47, 2]
ax.scatter(X, Y, Z)

X = points[48, 0]
Y = points[48, 1]
Z = points[48, 2]
ax.scatter(X, Y, Z)

# array_points=dslice.get('point{}'.format(len(dslice)-1))
# X=array_points[0,:]
# Y=array_points[1,:]
# Z=array_points[2,:]

# ax.plot_trisurf(X,Y,Z)
i = 47

array_points = dslice.get("point{}".format(i))
X = array_points[0, :]
Y = array_points[1, :]
Z = array_points[2, :]
ax.plot_trisurf(X, Y, Z, label="anomaly")

ax.view_init(30, -30)

plt.show()


# plot ctgr slices
def get_surfaces(fname, pinfo, case):

    os.chdir(
        "N:/vasospasm/"
        + pinfo
        + "/"
        + case
        + "/1-geometry/"
        + pinfo
        + "_"
        + case
        + "_segmentation_no_vti/Segmentations"
    )

    fname = "L_ICA_MCA.ctgr"
    with open(fname) as f:
        xml = f.read()
        root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    # find the branch of the tree which contains control points

    branch = root[1][0]
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

    return dsurfaces


def find_radius(center, coord_center):

    L = []
    for i in range(coord_center.shape[0]):
        L.append(np.linalg.norm(center - coord_center[:, i]))
    radius = max(L)

    Lrad = [L.index(x) for x in L if x > 0.9 * radius]
    coord_rad = np.asarray([coord_center[:, i] for i in Lrad])

    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.grid()
    # ax.scatter(coord_center[0,:], coord_center[1,:],
    #            coord_center[2,:] )

    # ax.plot(center[0],center[1],center[2],c='k',marker='x')

    Xrad = np.zeros((2, 3))
    Xrad[0] = center
    Xrad[1] = coord_rad[1]  # Plot the radius
    # ax.plot(Xrad[:,0],Xrad[:,1],Xrad[:,2])

    # ax.view_init(30,0)
    # plt.show()

    return radius


def find_center(coordinates_circle):

    x_mean = np.mean(coordinates_circle[:, 0])
    y_mean = np.mean(coordinates_circle[:, 1])
    z_mean = np.mean(coordinates_circle[:, 2])

    center = np.array([x_mean, y_mean, z_mean])

    return center


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
        + "_segmentation_no_vti/Segmentations"
    )

    fname = "L_ICA_MCA.ctgr"
    with open(fname) as f:
        xml = f.read()
        root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    # find the branch of the tree which contains control points

    branch = root[1][0]
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


dsurfaces = get_surfaces("L_ICA_MCA", "pt2", "baseline")
dcenter = get_center_radius("L_ICA_MCA", "pt2", "baseline")
Lrad = []
for i in range(20, len(dsurfaces) - 6):
    array_points = dsurfaces.get("surface{}".format(i))
    center, radius = dcenter.get("center{}".format(i))
    Lrad.append(radius)
    X = array_points[:, 0]
    Y = array_points[:, 1]
    Z = array_points[:, 2]

    # ax.plot_trisurf(X,Y,Z)

X_i = np.linspace(20, len(dsurfaces) - 6, len(dsurfaces) - 20 - 5)[:-1]
plt.plot(X_i, Lrad, label="radius")
plt.ylabel("radius (m)")
plt.title("plot of the radius in the curve of L_ICA | segmentation slices")
plt.show()

Lrad2 = []
for i in range(46, 55):
    array_points = dslice.get("point{}".format(i))
    center = find_center(array_points)
    radius = find_radius(center, array_points)

    Lrad2.append(radius)
    X = array_points[:, 0]
    Y = array_points[:, 1]
    Z = array_points[:, 2]

X_i = np.linspace(46, 55, 9)
plt.plot(X_i, Lrad2, label="radius")
plt.ylabel("radius (m)")
plt.title("plot of the radius in the curve of L_ICA | segmentation slices")
plt.show()

#%% Test polydata
#%% Test outlet


def plot_vessels_outlet(dpoints, i_vessel):

    name = dpoints.get("points{}".format(i_vessel))[0]
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

    Lfiles = data_file_baseline.zone_names
    new_name = change_name(name, Lfiles)

    print(Lfiles)


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
    "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit/" + filename
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
name = "pt7_baseline.lmca"
points = dp.get("points{}".format(11))[1]
name_vessel = dp.get("points{}".format(11))[0]


cx = data_file_baseline.zone(name).values("X")[:]
cy = data_file_baseline.zone(name).values("Y")[:]
cz = data_file_baseline.zone(name).values("Z")[:]
x_base = np.asarray(cx)
y_base = np.asarray(cy)
z_base = np.asarray(cz)

coordinates_fluent = np.array([x_base, y_base, z_base]).T

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

ax.grid(False)


ax.scatter(x_base, y_base, z_base, label="outlet")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=name_vessel)
ax.view_init(30, 60)
ax.legend()

plt.show()


import vtk
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
)


import pyvista as pv


pinfo = "pt2"
case = "baseline"
filename = "pt2_baseline_cycle2-1-02257.dat"
step = 1

logging.basicConfig(level=logging.DEBUG)

# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

tp.session.connect()
tp.new_layout()
frame = tp.active_frame()

dir_file = (
    "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit/" + filename
)

data_file = tp.data.load_fluent(
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

tp.macro.execute_command("$!RedrawAll")
tp.active_frame().plot().rgb_coloring.red_variable_index = 3
tp.active_frame().plot().rgb_coloring.green_variable_index = 3
tp.active_frame().plot().rgb_coloring.blue_variable_index = 13
tp.active_frame().plot().contour(0).variable_index = 3
tp.active_frame().plot().contour(1).variable_index = 4
tp.active_frame().plot().contour(2).variable_index = 5
tp.active_frame().plot().contour(3).variable_index = 6
tp.active_frame().plot().contour(4).variable_index = 7
tp.active_frame().plot().contour(5).variable_index = 8
tp.active_frame().plot().contour(6).variable_index = 9
tp.active_frame().plot().contour(7).variable_index = 10
tp.active_frame().plot().show_contour = True
tp.macro.execute_command("$!RedrawAll")
tp.active_frame().plot(PlotType.Cartesian3D).show_slices = True
slices = frame.plot().slices()
slices.contour.show = True
frame = tp.active_frame()
plot = frame.plot()
plot.show_slices = True
slices = plot.slices(0)
slices.show = True

name = "region"

cx = data_file.zone(name).values("X")[:]
cy = data_file.zone(name).values("Y")[:]
cz = data_file.zone(name).values("Z")[:]
x_base = np.asarray(cx)
y_base = np.asarray(cy)
z_base = np.asarray(cz)
points = np.array([x_base, y_base, z_base]).T

# Using pyvista polydata

point_cloud = pv.PolyData(points)
# surf = point_cloud.delaunay_2d(alpha=0.90,progress_bar=True)
# surf.plot(cpos="xy", show_edges=True)
# #Difining the surface

# surface = point_cloud.delaunay_2d(alpha=point_cloud.length/15,progress_bar=True).extract_surface().triangulate().clean()
# point_cloud_numpy_array has shape (N, 3) where N is the number of points


surface = (
    point_cloud.delaunay_2d(progress_bar=True).extract_surface().triangulate().clean()
)
areas = surface.compute_cell_sizes().cell_arrays["Area"]
final_surface = (
    surface.extract_cells(areas < 8 * 1e-8).extract_surface().triangulate().clean()
)

final_surface.plot()


# surface.plot()
# #Shell

# shell = surface.extract_geometry()
# shell.plot()

# Defining the control points

# grid = pv.StructuredGrid(x_base, y_base, z_base)
# grid.plot()


def get_spline_points(fname, step):

    with open(fname) as f:
        xml = f.read()
        root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    # find the branch of the tree which contains control points
    branch = root[1][0][0][1]

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


if pinfo == "pt2":
    folder = "_segmentation_no_vti"
else:
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

# dpoints_updated={}
# i=0
# for file in onlyfiles:
#     points_ref=get_spline_points(file, step)
#     points_ref_pv=pv.PolyData(points_ref)
#     selected = points_ref_pv.select_enclosed_points(surface,check_surface=False)
#     pts = points_ref_pv.extract_points(selected['SelectedPoints'].view(bool),
#                             adjacent_cells=False)
#     # pl = pv.Plotter()
#     # _ = pl.add_mesh(surface, style='wireframe')
#     # _ = pl.add_points(pts, color='r')
#     dpoints_updated['points{}'.format(i)]=pts
#     i+=1
# # cloud.plot()
dpoints_updated_arrays = {}
for i in range(len(dpoints_updated)):
    source = dpoints_updated.get("points{}".format(i))
    n = source.number_of_cells
    array_points = np.zeros((n, 3))
    for j in range(n):
        array_points[j] = source.cell_points(j)
    dpoints_updated_arrays["points{}".format(i)] = array_points


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# for i in range(len(dpoints_updated_arrays)):
vessel_points = dpoints_updated_arrays.get("points{}".format(9))
x = vessel_points[:, 0]
y = vessel_points[:, 1]
z = vessel_points[:, 2]

ax.scatter(x, y, z)

plt.show()


# surf=cloud.delaunay_2d()

# surf.plot()
