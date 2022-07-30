# -*- coding: utf-8 -*-

"""
Created on Wed Jul 13 11:50:26 2022

@author: GALADRIEL_GUEST
"""

# Script whose main function returns the actual divided geometry for the variation 2
# --> No left A1


# %% Imports

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
from scipy.interpolate import interp1d
import logging
import skg
from skg import nsphere
import pickle
from tqdm import tqdm


# %% Functions


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


def create_dpoint(pinfo, case, step):
    """


    Parameters
    ----------
    pinfo : str, example : 'pt2' , 'vsp7'
    case : str, 'baseline' or 'vasospasm'

    Returns
    -------
    dpoint_i : dict of all the control points for the vessels of the patient

    """

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
    i = 0
    dpoint_i = {}
    for file in onlyfiles:
        filename = file[:-4]
        dpoint_i["points{}".format(i)] = filename, get_spline_points(file, step)
        i += 1
    return dpoint_i


def division_ICA(pinfo, case, step):
    """


    Parameters
    ----------
    pinfo : str, example : 'pt2' , 'vsp7'
    case : str, 'baseline' or 'vasospasm'

    Returns
    -------
    dpoints_divided : dict of all the control points for the vessels of the patient,
    with the ICA_MCA --> ICA & MCA for left and right.

    """

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
        if "L_ACA" in files:
            points_LACA = get_spline_points(files, step)
        if "R_ACA" in files:
            points_RACA = get_spline_points(files, step)
        if "L_ICA_MCA" in files:
            points_LICAMCA = get_spline_points(files, step)
        if "R_ICA_MCA" in files:
            points_RICAMCA = get_spline_points(files, step)

    # Manual division on the left side | No Left ACA A1 to find the
    # intersection

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid()
    ax.scatter(
        points_LICAMCA[:, 0],
        points_LICAMCA[:, 1],
        points_LICAMCA[:, 2],
        label="LEFT ICA",
    )
    ax.scatter(
        points_RICAMCA[:, 0],
        points_RICAMCA[:, 1],
        points_RICAMCA[:, 2],
        label="RIGHT ICA ",
    )
    ax.scatter(
        points_LACA[:, 0], points_LACA[:, 1], points_LACA[:, 2], label="LEFT ACA "
    )
    ax.scatter(
        points_RACA[:, 0], points_RACA[:, 1], points_RACA[:, 2], label="RIGHT ACA "
    )
    ax.view_init(30, 90)
    ax.legend()

    step_point = int((len(points_LICAMCA)) / (len(points_LICAMCA) / 2))
    X, Y, Z = points_LICAMCA[:, 0], points_LICAMCA[:, 1], points_LICAMCA[:, 2]
    L = np.linspace(0, len(points_LICAMCA) - 1, 20)
    Lind = [int(np.floor(x)) for x in L]
    print(Lind)
    for ik in Lind:
        annotation = "point {}".format(ik)
        x, y, z = list(zip(X, Y, Z))[ik]
        ax.text(x, y, z, annotation)

    plt.show()

    print("\n")
    print("## Select separation point ##   " + "LICAMCA" + "\n")
    for i in range(len(points_LICAMCA)):
        print("   ", i, " : point ", i)

    target = int(input("-->  "))

    points_LICA = points_LICAMCA[:target]
    points_LMCA = points_LICAMCA[target:]

    # Automatic method for the Right side | Finding the minimum of norms to
    # find the intersection with ACA

    rtarget = [points_RACA[0], points_RACA[points_RACA.shape[0] - 1]]
    rnorms_end = []
    rnorms_start = []
    for i in range(points_RICAMCA.shape[0]):
        norm_end = np.linalg.norm(rtarget[1] - points_RICAMCA[i])
        norm_start = np.linalg.norm(rtarget[0] - points_RICAMCA[i])
        rnorms_end.append(norm_end)
        rnorms_start.append(norm_start)

    Ltot_norms = rnorms_end + rnorms_start
    rmini = np.min(Ltot_norms)
    rimin = Ltot_norms.index(rmini)
    if rimin > len(rnorms_end):
        rimin -= len(rnorms_end)

    points_RICA = points_RICAMCA[:rimin]
    points_RMCA = points_RICAMCA[rimin:]

    # FINAL VISUALIZATION

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid()

    ax.scatter(
        points_LICA[:, 0], points_LICA[:, 1], points_LICA[:, 2], label="LEFT ICA"
    )
    ax.scatter(
        points_RICA[:, 0], points_RICA[:, 1], points_RICA[:, 2], label="RIGHT ICA "
    )
    ax.scatter(
        points_RMCA[:, 0], points_RMCA[:, 1], points_RMCA[:, 2], label="RIGHT MCA"
    )
    ax.scatter(
        points_LMCA[:, 0], points_LMCA[:, 1], points_LMCA[:, 2], label="LEFT MCA"
    )
    ax.scatter(
        points_LACA[:, 0], points_LACA[:, 1], points_LACA[:, 2], label="LEFT ACA "
    )
    ax.scatter(
        points_RACA[:, 0], points_RACA[:, 1], points_RACA[:, 2], label="RIGHT ACA "
    )

    ax.view_init(30, 90)
    ax.legend()
    plt.show()

    dpoints_divided = {}
    k = 0
    if points_LICA.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "L_ICA", points_LICA
        k += 1
    if points_RICA.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "R_ICA", points_RICA
        k += 1
    if points_LMCA.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "L_MCA", points_LMCA
        k += 1
    if points_RMCA.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "R_MCA", points_RMCA
        k += 1

    return dpoints_divided


def division_ACAs(pinfo, case, step):
    """


    Parameters
    ----------
    pinfo : str. example : 'pt2'
    case : str. example : 'baseline'
    vessel : .pth file of the vessel.

    Returns
    -------
    dpoints_divided :dictionary of the control points

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
        if "Acom" in files:
            points_acom = get_spline_points(files, step)
        if "L_ACA" in files:
            points_laca = get_spline_points(files, step)
        if "R_ACA" in files:
            points_raca = get_spline_points(files, step)

    # VISUALIZATION

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(points_laca[:, 0], points_laca[:, 1], points_laca[:, 2], label="L_ACA")
    ax.scatter(points_raca[:, 0], points_raca[:, 1], points_raca[:, 2], label="R_ACA")
    ax.legend()
    # ax.view_init(30,30)

    # DIVISION L_ACA INTO LACA A1 & LACA A2

    points_LA2 = points_laca

    # DIVISION R_ACA INTO RACA A1 & RACA A2

    target = [points_acom[0], points_acom[points_acom.shape[0] - 1]]
    lnorms_end = []
    lnorms_start = []

    for i in range(points_raca.shape[0]):
        # Norm between first/last  LACA points and LICAMCA points
        norm_end = np.linalg.norm(target[1] - points_raca[i])
        norm_start = np.linalg.norm(target[0] - points_raca[i])
        lnorms_end.append(norm_end)
        lnorms_start.append(norm_start)
    # Min of the two lists
    Ltot_norms = lnorms_end + lnorms_start

    lmini = np.min(Ltot_norms)
    limin = Ltot_norms.index(lmini)
    if limin > len(lnorms_end):
        limin -= len(lnorms_end)

    points_RA1 = points_raca[limin:]
    points_RA2 = points_raca[:limin]

    # Final Visualization

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(
        points_acom[:, 0], points_acom[:, 1], points_acom[:, 2], c="g", label="Acom"
    )
    ax.scatter(
        points_LA2[:, 0], points_LA2[:, 1], points_LA2[:, 2], c="b", label="LACA A2"
    )
    ax.scatter(
        points_RA1[:, 0], points_RA1[:, 1], points_RA1[:, 2], c="k", label="RACA A1"
    )
    ax.scatter(
        points_RA2[:, 0], points_RA2[:, 1], points_RA2[:, 2], c="r", label="RACA A2"
    )

    ax.view_init(30, 30)
    ax.legend()

    dpoints_divided = {}
    k = 0
    # if points_LA1.shape[0] != 0:
    #     dpoints_divided["points{}".format(k)] = "LA1", points_LA1
    #     k += 1
    if points_LA2.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "LA2", points_LA2
        k += 1
    if points_RA1.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "RA1", points_RA1
        k += 1
    if points_RA2.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "RA2", points_RA2
        k += 1

    return dpoints_divided


def manual_division(pinfo, case, vessel, step):
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
    print(vessel)
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
# L : list of two str, being 'auto' if there is a left/right Pcom,
# 'manual' if not

#     """

#     dpoints = create_dpoint(pinfo, case)
#     L = ["manual", "manual"]
#     for i in range(len(dpoints)):
#         if "L_Pcom" in dpoints.get("points{}".format(i))[0]:
#             L[0] = "auto"
#         if "R_Pcom" in dpoints.get("points{}".format(i))[0]:
#             L[1] = "auto"
#     return L


def division_A(pinfo, case, step):
    """


    Parameters
    ----------
    pinfo : str, example : 'pt2' , 'vsp7'
    case : str, 'baseline' or 'vasospasm'

    Returns
    -------
    dpoints_divided : dict of the control points for every vessel

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
        if "Acom" in files:
            points_Acom = get_spline_points(files, step)
        if "L_ACA" in files:
            points_LACA = get_spline_points(files, step)
        if "R_ACA" in files:
            points_RACA = get_spline_points(files, step)

    # Assuming that Acom is well oriented from left to right

    ltarget = points_Acom[points_Acom.shape[0] - 1]
    rtarget = points_Acom[0]

    lnorms = []
    for i in range(points_LACA.shape[0]):
        lnorm = np.linalg.norm(ltarget - points_LACA[i])
        lnorms.append(lnorm)
    lmini = np.min(lnorms)
    limin = lnorms.index(lmini)
    points_LA1 = points_LACA[:limin]
    points_LA2 = points_LACA[limin:]
    rnorms = []
    for i in range(points_RACA.shape[0]):
        rnorm = np.linalg.norm(rtarget - points_RACA[i])
        rnorms.append(rnorm)
    rmini = np.min(rnorms)
    imin = rnorms.index(rmini)

    points_RA1 = points_RACA[:imin]
    points_RA2 = points_RACA[imin:]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid()

    ax.scatter(points_Acom[:, 0], points_Acom[:, 1], points_Acom[:, 2], label="Acom")
    ax.scatter(
        points_RA1[:, 0], points_RA1[:, 1], points_RA1[:, 2], label="RIGHT ACA A1"
    )
    ax.scatter(
        points_RA2[:, 0], points_RA2[:, 1], points_RA2[:, 2], label="RIGHT ACA A2"
    )
    ax.scatter(
        points_LA1[:, 0], points_LA1[:, 1], points_LA1[:, 2], label="LEFT ACA A1"
    )
    ax.scatter(
        points_LA2[:, 0], points_LA2[:, 1], points_LA2[:, 2], label="LEFT ACA A2"
    )

    ax.view_init(30, 90)
    ax.legend()
    plt.show()

    dpoints_divided = {}
    k = 0
    if points_LA1.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "L_A1", points_LA1
        k += 1
    if points_LA2.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "L_A2", points_LA2
        k += 1
    if points_RA1.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "R_A1", points_RA1
        k += 1
    if points_RA2.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "R_A2", points_RA2
        k += 1

    return dpoints_divided


def division_P_bas(pinfo, case, step):
    """


    Parameters
    ----------
    pinfo : str, example : 'pt2' , 'vsp7'
    case : str, 'baseline' or 'vasospasm'

    Returns
    -------
    dpoints_divided : dict of the control points for every vessel

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

        # If one of the PCA is merged with the basilar : separation

        if "BAS_PCA" in files:
            points_bas_pca = get_spline_points(files, step)
            side_bas = files[0]

            for subfile in onlyfiles:

                if side_bas == "L":
                    if "R_PCA" in subfile:
                        points_target = get_spline_points(subfile, step)

                else:
                    if "L_PCA" in subfile:
                        points_target = get_spline_points(subfile, step)
            # NEW METHOD - Works whatever the direction of the vessel

            target = [points_target[0], points_target[points_target.shape[0] - 1]]
            lnorms_end = []
            lnorms_start = []

            for i in range(points_bas_pca.shape[0]):
                # Norm between first/last  LACA points and LICAMCA points
                norm_end = np.linalg.norm(target[1] - points_bas_pca[i])
                norm_start = np.linalg.norm(target[0] - points_bas_pca[i])
                lnorms_end.append(norm_end)
                lnorms_start.append(norm_start)
            # Min of the two lists
            Ltot_norms = lnorms_end + lnorms_start

            lmini = np.min(Ltot_norms)
            limin = Ltot_norms.index(lmini)
            if limin > len(lnorms_end):
                limin -= len(lnorms_end)

            # DIVISION BAS & PCA

            points_pca = points_bas_pca[limin:]
            points_bas = points_bas_pca[:limin]
            print(points_bas.shape)
            print("\n")
            print(points_pca.shape)

            if side_bas == "L":
                side_vessel = "R"
            else:
                side_vessel = "L"

            print(side_vessel)
            print(side_bas)
            for subfile in onlyfiles:

                if side_vessel + "_Pcom" in subfile:
                    points_Pcom = get_spline_points(subfile, step)
                    print(subfile)
                elif side_bas + "_Pcom" in subfile:
                    points_bas_Pcom = get_spline_points(subfile, step)
                    print(subfile)

                # elif side_vessel + '_PCA' in subfile:
                #     points_pca=get_spline_points(subfile,step)

            # DIVISIN P1 P2 ON BASILAR SIDE

            # Definition of the target points (take the first and last to be
            # direction independent)
            target = [points_Pcom[0], points_Pcom[points_Pcom.shape[0] - 1]]
            target_bas = [
                points_bas_Pcom[0],
                points_bas_Pcom[points_bas_Pcom.shape[0] - 1],
            ]
            print(target_bas)
            bas_norms_start = []
            bas_norms_end = []
            for i in range(points_pca.shape[0]):
                norm_start = np.linalg.norm(target_bas[0] - points_pca[i])
                norm_end = np.linalg.norm(target_bas[1] - points_pca[i])

                bas_norms_start.append(norm_start)
                bas_norms_end.append(norm_end)

            Ltot_norms = bas_norms_end + bas_norms_start
            lmini = np.min(Ltot_norms)
            limin = Ltot_norms.index(lmini)
            print("Len : ", bas_norms_start)
            print("limin : ", limin)
            if limin > len(bas_norms_end):
                limin -= len(bas_norms_end)

            points_bas_P1 = points_pca[:limin]
            points_bas_P2 = points_pca[limin:]

            print("P1 : ", points_bas_P1.shape)
            print("P2 : ", points_bas_P2)

            # SEPARATION P1 P2 NOT ON THE BASILAR SIDE

            nb_norms_start = []
            nb_norms_end = []
            for i in range(points_target.shape[0]):
                norm_start = np.linalg.norm(target[0] - points_target[i])
                norm_end = np.linalg.norm(target[1] - points_target[i])

                nb_norms_start.append(norm_start)
                nb_norms_end.append(norm_end)

            Ltot_norms = nb_norms_end + nb_norms_start
            lmini = np.min(Ltot_norms)
            limin = Ltot_norms.index(lmini)

            if limin > len(nb_norms_end):
                limin -= len(nb_norms_end)

            points_P1 = points_target[:limin]
            points_P2 = points_target[limin:]

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.grid()
            ax.scatter(
                points_bas[:, 0], points_bas[:, 1], points_bas[:, 2], label="basilar"
            )

            ax.scatter(
                points_bas_Pcom[:, 0],
                points_bas_Pcom[:, 1],
                points_bas_Pcom[:, 2],
                label=" basilar side Pcom",
            )
            ax.scatter(
                points_Pcom[:, 0],
                points_Pcom[:, 1],
                points_Pcom[:, 2],
                label="non basilar side Pcom",
            )
            ax.scatter(
                points_bas_P1[:, 0],
                points_bas_P1[:, 1],
                points_bas_P1[:, 2],
                label="basilar side P1",
            )
            ax.scatter(
                points_bas_P2[:, 0],
                points_bas_P2[:, 1],
                points_bas_P2[:, 2],
                label="basilar side P2",
            )
            ax.scatter(
                points_P1[:, 0],
                points_P1[:, 1],
                points_P1[:, 2],
                label="Non Basilar side P1",
            )
            ax.scatter(
                points_P2[:, 0],
                points_P2[:, 1],
                points_P2[:, 2],
                label="Non Basilar side P2",
            )

            ax.legend()
            plt.show()

            dpoints_divided = {}
            k = 0
            if points_bas.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = side_bas + "_BAS", points_bas
                k += 1
            if points_bas_P1.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = side_bas + "_P1", points_bas_P1
                k += 1
            if points_bas_P2.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = side_bas + "_P2", points_bas_P2
                k += 1
            if points_P1.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = side_vessel + "_P1", points_P1
                k += 1
            if points_P2.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = side_vessel + "_P2", points_P2
                k += 1

            return dpoints_divided

        # else :

        #     for subfile in onlyfiles :
        #         if "Acom" in subfile:
        #             points_Acom = get_spline_points(subfile,step)
        #         if "L_PCA" in subfile:
        #             points_LACA = get_spline_points(subfile,step)
        #         if "R_PCA" in subfile:
        #             points_RACA = get_spline_points(subfile,step)

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
        #         dpoints_divided["points{}".format(k)] = "L_P1", points_LA1
        #         k += 1
        #     if points_LA2.shape[0] != 0:
        #         dpoints_divided["points{}".format(k)] = "L_P2", points_LA2
        #         k += 1
        #     if points_RA1.shape[0] != 0:
        #         dpoints_divided["points{}".format(k)] = "R_P1", points_RA1
        #         k += 1
        #     if points_RA2.shape[0] != 0:
        #         dpoints_divided["points{}".format(k)] = "R_P2", points_RA2
        #         k += 1

    return dpoints_divided


def add_divided_arteries(dpoint_i, dpoints_div):
    """
    Parameters
    ----------
    dpoint_i : dict of the control points, ICA_MCA release.
    dpoints_div : dict of the ICA & MCA control points.

    Returns
    -------
    dpoint_i : dict of the control points, ICA_MCA & ICA & MCA release.

    """

    counterbalance = 0
    for j in range(len(dpoint_i), len(dpoint_i) + len(dpoints_div)):
        k = j - len(dpoint_i) + counterbalance
        dpoint_i["points{}".format(j)] = (
            dpoints_div.get("points{}".format(k))[0],
            dpoints_div.get("points{}".format(k))[1],
        )
        counterbalance += 1
    return dpoint_i


def delete_old_arteries(dpoint_i):
    """
    Parameters
    ----------
    dpoint_i : dict of the control point, ICA_MCA & ICA & MCA release.

    Returns
    -------
    dpoint_i : dict of the control points, ICA & MCA without ICA_MCA release.
    indices : list of the indices which are kept after deleting the fusionned arteries.

    """

    I_supp = []
    for j in range(len(dpoint_i)):

        if "ICA_MCA" in dpoint_i.get("points{}".format(j))[0]:

            # del dpoint_i["points{}".format(j)]
            I_supp.append(j)

        if (
            dpoint_i.get("points{}".format(j))[0] is not None
            and "L_ACA" in dpoint_i.get("points{}".format(j))[0]
        ):

            # del dpoint_i["points{}".format(j)]
            I_supp.append(j)

        if (
            dpoint_i.get("points{}".format(j))[0] is not None
            and "R_ACA" in dpoint_i.get("points{}".format(j))[0]
        ):

            # del dpoint_i["points{}".format(j)]
            I_supp.append(j)

        if (
            dpoint_i.get("points{}".format(j))[0] is not None
            and "L_PCA" in dpoint_i.get("points{}".format(j))[0]
        ):

            # del dpoint_i["points{}".format(j)]
            I_supp.append(j)

        if (
            dpoint_i.get("points{}".format(j))[0] is not None
            and "R_PCA" in dpoint_i.get("points{}".format(j))[0]
        ):
            I_supp.append(j)

        if (
            dpoint_i.get("points{}".format(j))[0] is not None
            and "aneurysm" in dpoint_i.get("points{}".format(j))[0]
        ):
            I_supp.append(j)

    for i in I_supp:
        del dpoint_i["points{}".format(i)]
    indices = [i for i in range(len(dpoint_i) + len(I_supp)) if i not in I_supp]
    return dpoint_i, indices


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


def createfinal_dicts(dpoint_i, indices):
    j = 0
    dpoints = {}
    dvectors = {}
    for i in indices:
        filename, points = (
            dpoint_i.get("points{}".format(i))[0],
            dpoint_i.get("points{}".format(i))[1],
        )
        dpoints["points{}".format(j)] = filename, points
        dvectors["vectors{}".format(j)] = filename, calculate_normal_vectors(points)
        j += 1
    return dpoints, dvectors


# %% Main


def _main_(pinfo, case, step):

    dpoint_i = create_dpoint(pinfo, case, step)

    # Step 2# Divide

    dpoints_divI = division_ICA(pinfo, case, step)
    dpoints_divACA = division_ACAs(pinfo, case, step)
    dpoints_divPCA = division_P_bas(pinfo, case, step)

    dpoints = dpoint_i.copy()

    # Step 3# Add

    dpoints = add_divided_arteries(dpoints, dpoints_divI)
    dpoints = add_divided_arteries(dpoints, dpoints_divACA)
    dpoints = add_divided_arteries(dpoints, dpoints_divPCA)

    # Delete

    dpoints, indices = delete_old_arteries(dpoints)

    dpoints, dvectors = createfinal_dicts(dpoints, indices)

    return dpoints, dvectors
