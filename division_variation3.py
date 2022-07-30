# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:16:42 2022

@author: GALADRIEL_GUEST
"""


# Script whose main function returns the actual divided geometry for the first variation
# --> left fetal PCA (no left P1)

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

    ltarget = [points_LACA[0], points_LACA[points_LACA.shape[0] - 1]]
    rtarget = [points_RACA[0], points_RACA[points_RACA.shape[0] - 1]]
    lnorms_end = []
    lnorms_start = []

    for i in range(points_LICAMCA.shape[0]):
        # Norm between first/last  LACA points and LICAMCA points
        norm_end = np.linalg.norm(ltarget[1] - points_LICAMCA[i])
        norm_start = np.linalg.norm(ltarget[0] - points_LICAMCA[i])

        lnorms_end.append(norm_end)
        lnorms_start.append(norm_start)

    # Min of the two lists
    Ltot_norms = lnorms_end + lnorms_start
    lmini = np.min(Ltot_norms)
    limin = Ltot_norms.index(lmini)
    if limin > len(lnorms_end):
        limin -= len(lnorms_end)

    points_LICA = points_LICAMCA[:limin]
    points_LMCA = points_LICAMCA[limin:]

    # Same Method for the right side

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

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid()

    ax.scatter(
        points_LICA[:, 0], points_LICA[:, 1], points_LICA[:, 2], c="b", label="LEFT ICA"
    )
    ax.scatter(
        points_RICA[:, 0],
        points_RICA[:, 1],
        points_RICA[:, 2],
        c="k",
        label="RIGHT ICA ",
    )
    ax.scatter(
        points_RMCA[:, 0],
        points_RMCA[:, 1],
        points_RMCA[:, 2],
        c="r",
        label="RIGHT MCA",
    )
    ax.scatter(
        points_LMCA[:, 0], points_LMCA[:, 1], points_LMCA[:, 2], c="g", label="LEFT MCA"
    )
    ax.scatter(
        points_LACA[:, 0], points_LACA[:, 1], points_LACA[:, 2], label="LEFT ACA "
    )
    ax.scatter(
        points_RACA[:, 0], points_RACA[:, 1], points_RACA[:, 2], label="RIGHT ACA "
    )

    ax.view_init(30, 60)
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


# def get_method_div_A(pinfo, case,step):
#     """


#     Parameters
#     ----------
#     pinfo : str, example : 'pt2' , 'vsp7'
#     case : str, 'baseline' or 'vasospasm'

#     Returns
#     -------
#     str : 'auto' if there is an Acom, 'manual' if not

#     """
#     dpoints = create_dpoint(pinfo, case,step)

#     for i in range(len(dpoints)):
#         if "Acom" in dpoints.get("points{}".format(i))[0]:
#             return "auto"
#     return "manual"


def division_LP(pinfo, case, vessel, step):
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

    if vessel[0] == "L":
        other_side = "R"
    else:
        other_side = "R"

    for file in glob.glob("*.pth"):
        onlyfiles.append(file)
    for files in onlyfiles:
        if vessel in files:
            points_vessel = get_spline_points(files, step)
        if "BAS_PCA" in files:
            points_bas = get_spline_points(files, step)
        if other_side + "_Pcom" in files:
            points_pcom = get_spline_points(files, step)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(
        points_bas[:, 0],
        points_bas[:, 1],
        points_bas[:, 2],
        label=other_side + "_BAS_PCA",
    )
    ax.scatter(
        points_pcom[:, 0],
        points_pcom[:, 1],
        points_pcom[:, 2],
        label=other_side + "_Pcom",
    )
    X, Y, Z = points_vessel[:, 0], points_vessel[:, 1], points_vessel[:, 2]
    ax.plot(X, Y, Z, "--")
    ax.plot(X, Y, Z, "o", label=vessel)

    ax.legend()
    plt.title(vessel)
    # i_point=0
    step_point = int((len(points_vessel)) / (len(points_vessel) / 2))

    L = np.linspace(0, len(points_vessel) - 1, 20)
    Lind = [int(np.floor(x)) for x in L]
    print(Lind)
    for ik in Lind:

        annotation = "point {}".format(ik)
        x, y, z = list(zip(X, Y, Z))[ik]
        ax.text(x, y, z, annotation)

    # for  x, y, z in zip(X, Y, Z):
    #     annotation='point {}'.format(i_point)
    #     ax.text(x, y, z, annotation)
    #     i_point+=5
    plt.show()

    print("\n")
    print("## Select separation point ##   " + vessel[:-4] + "\n")
    for i in range(len(points_vessel)):
        print("   ", i, " : point ", i)

    target = int(input("-->  "))

    points_1 = points_vessel[target:]
    points_2 = points_vessel[:target]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(
        points_bas[:, 0],
        points_bas[:, 1],
        points_bas[:, 2],
        label=other_side + "_BAS_PCA",
    )
    ax.scatter(
        points_pcom[:, 0],
        points_pcom[:, 1],
        points_pcom[:, 2],
        label=other_side + "_Pcom",
    )
    ax.scatter(
        points_1[:, 0], points_1[:, 1], points_1[:, 2], label=other_side + "PCA P2"
    )
    ax.scatter(
        points_2[:, 0], points_2[:, 1], points_2[:, 2], label=other_side + "_Pcom"
    )

    ax.legend()

    dpoints_divided = {}
    k = 0
    if points_1.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "LP2", points_1
        k += 1
    if points_2.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "L_Pcom", points_2
        k += 1

    return dpoints_divided


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

    target = [points_Acom[0], points_Acom[points_Acom.shape[0] - 1]]

    Lnorms_start = []
    Lnorms_end = []
    for i in range(points_LACA.shape[0]):
        lnorm_start = np.linalg.norm(target[0] - points_LACA[i])
        lnorm_end = np.linalg.norm(target[1] - points_LACA[i])

        Lnorms_start.append(lnorm_start)
        Lnorms_end.append(lnorm_end)
    Lnorms_tot = Lnorms_end + Lnorms_start

    lmini = np.min(Lnorms_tot)
    limin = Lnorms_tot.index(lmini)
    if limin > len(Lnorms_end):
        limin -= len(Lnorms_end)

    points_LA1 = points_LACA[:limin]
    points_LA2 = points_LACA[limin:]

    Rnorms_start = []
    Rnorms_end = []
    for i in range(points_RACA.shape[0]):
        lnorm_start = np.linalg.norm(target[0] - points_RACA[i])
        lnorm_end = np.linalg.norm(target[1] - points_RACA[i])

        Rnorms_start.append(lnorm_start)
        Rnorms_end.append(lnorm_end)
    Rnorms_tot = Lnorms_end + Lnorms_start

    rmini = np.min(Rnorms_tot)
    rimin = Rnorms_tot.index(rmini)
    if rimin > len(Rnorms_end):
        rimin -= len(Rnorms_end)

    points_RA1 = points_RACA[:limin]
    points_RA2 = points_RACA[limin:]

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


def new_division_P_bas(pinfo, case, step):
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

                if side_bas + "_Pcom" in subfile:
                    points_target = get_spline_points(subfile, step)

                if "Pcom_PCA" in subfile:
                    points_otherside = get_spline_points(subfile, step)

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

            # DIVISION BAS/P1 AND P2

            points_P2 = points_bas_pca[limin:]
            points_basP1 = points_bas_pca[:limin]

            # DIVISION BAS & P1

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.grid(False)
            ax.scatter(
                points_otherside[:, 0],
                points_otherside[:, 1],
                points_otherside[:, 2],
                label="PCA_Pcom",
            )

            ax.scatter(
                points_P2[:, 0], points_P2[:, 1], points_P2[:, 2], c="b", label="P2"
            )
            ax.scatter(
                points_target[:, 0],
                points_target[:, 1],
                points_target[:, 2],
                c="g",
                label=side_bas + "_Pcom",
            )
            X, Y, Z = points_basP1[:, 0], points_basP1[:, 1], points_basP1[:, 2]
            # ax.scatter(X,Y,Z,'--')
            ax.scatter(X, Y, Z, "o", label="BAS + P1")

            ax.legend()
            plt.title("BAS + P1 Division")
            # i_point=0

            L = np.linspace(0, points_basP1.shape[0] - 1, 20)
            Lind = [int(np.floor(x)) for x in L]
            print(Lind)
            for ik in Lind:

                annotation = "point {}".format(ik)
                x, y, z = list(zip(X, Y, Z))[ik]
                ax.text(x, y, z, annotation)

            # for  x, y, z in zip(X, Y, Z):
            #     annotation='point {}'.format(i_point)
            #     ax.text(x, y, z, annotation)
            #     i_point+=5
            plt.show()

            print("\n")
            print("## Select separation point ##   " + "BAS_P1" + "\n")
            for i in range(len(points_basP1)):
                print("   ", i, " : point ", i)

            target = int(input("-->  "))

            points_bas = points_basP1[:target]
            points_P1 = points_basP1[target:]

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.grid()

            ax.scatter(
                points_target[:, 0],
                points_target[:, 1],
                points_target[:, 2],
                c="k",
                label=" basilar side Pcom",
            )
            ax.scatter(
                points_bas[:, 0],
                points_bas[:, 1],
                points_bas[:, 2],
                c="g",
                label="Basilar",
            )
            ax.scatter(
                points_P1[:, 0], points_P1[:, 1], points_P1[:, 2], c="b", label="PCA P1"
            )
            ax.scatter(
                points_P2[:, 0], points_P2[:, 1], points_P2[:, 2], c="r", label="PCA P2"
            )

            ax.legend()
            plt.show()

            dpoints_divided = {}
            k = 0
            if points_bas.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = side_bas + "_BAS", points_bas
                k += 1
            if points_P1.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = side_bas + "_P1", points_P1
                k += 1
            if points_P2.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = side_bas + "_P2", points_P2
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

        if "BAS" in files:
            points_bas_pca = get_spline_points(files, step)
            side_bas = files[0]
            for subfile in onlyfiles:

                if side_bas == "L":
                    if "R_Pcom_PCA" in subfile:
                        points_target = get_spline_points(subfile, step)

                else:
                    if "L_Pcom_PCA" in subfile:
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
            print(Ltot_norms)
            lmini = np.min(Ltot_norms)
            limin = Ltot_norms.index(lmini)
            if limin > len(lnorms_end):
                limin -= len(lnorms_end)

            # DIVISION BAS & PCA

            points_pca = points_bas_pca[:limin]
            points_bas = points_bas_pca[limin:]
            print(points_bas.shape)
            print("\n")
            print(points_pca.shape)

            points_bas_pca = points_pca

            for subfile in onlyfiles:

                if side_bas + "_Pcom" in subfile:
                    points_bas_Pcom = get_spline_points(subfile, step)
                    print("RPCOM : ", points_bas_Pcom)

            # Division between P1 and P2 on the basilar side

            target_bas = [
                points_bas_Pcom[0],
                points_bas_Pcom[points_bas_Pcom.shape[0] - 1],
            ]
            Lnorms_start = []
            Lnorms_end = []
            for i in range(points_pca.shape[0]):
                lnorm_start = np.linalg.norm(target_bas[0] - points_bas_pca[i])
                lnorm_end = np.linalg.norm(target_bas[1] - points_bas_pca[i])

                Lnorms_start.append(lnorm_start)
                Lnorms_end.append(lnorm_end)

            Lnorms_tot = Lnorms_start + Lnorms_end
            norm_min_pca = np.min(Lnorms_tot)
            index_min_pca = lnorms.index(norm_min_pca)

            if index_min_pca > len(Lnorms_start):
                index_min_pca -= len(Lnorms_start)
            points_bas_P1 = points_pca[:index_min_pca]
            points_bas_P2 = points_pca[index_min_pca:]

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

            return dpoints_divided

    return dpoints_divided


# def divide_LP(pinfo, case, step):


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

    # Step 2# CREATE NEW DIVIDED VESSELS

    dpoints_divI = division_ICA(pinfo, case, step)
    dpoints_divACA = division_A(pinfo, case, step)
    dpoints_divRPCA = new_division_P_bas(pinfo, case, step)
    dpoints_divLP = division_LP(pinfo, case, "L_Pcom_PCA.pth", step)

    dpoints = dpoint_i.copy()

    # Step 3# DELETE THE OLD VESSELS

    dpoints = add_divided_arteries(dpoints, dpoints_divI)

    dpoints = add_divided_arteries(dpoints, dpoints_divACA)

    dpoints = add_divided_arteries(dpoints, dpoints_divRPCA)
    dpoints = add_divided_arteries(dpoints, dpoints_divLP)

    dpoints, indices = delete_old_arteries(dpoints)

    dpoints, dvectors = createfinal_dicts(dpoints, indices)

    return dpoints, dvectors
