# -*- coding: utf-8 -*-

"""
Created on Wed Jul 13 11:50:26 2022

@author: GALADRIEL_GUEST
"""

# Script whose main function returns the actual divided geometry for the variation 2
# --> No Acom


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
import importlib

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

#%% Import intern module

os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

import geometry_slice as geom

importlib.reload(geom)


# %% Functions


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

    # LOAD .pth files (Control points)

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
            points_LACA = geom.get_spline_points(files, step)
        if "R_ACA" in files:
            points_RACA = geom.get_spline_points(files, step)
        if "L_ICA_MCA" in files:
            points_LICAMCA = geom.get_spline_points(files, step)
        if "R_ICA_MCA" in files:
            points_RICAMCA = geom.get_spline_points(files, step)

    # LOAD .ctgr files (center, radius)

    pathctgr = (
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
    os.chdir(pathctgr)

    filesctgr = []
    for file in glob.glob("*.ctgr"):
        filesctgr.append(file)
    for files in filesctgr:
        if "L_ACA" in files:
            center_LACA = geom.get_center_radius_ulti(files, pinfo, case)
        if "R_ACA" in files:
            center_RACA = geom.get_center_radius_ulti(files, pinfo, case)

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

    # Udpdate the index of the separation point, depending on the direction of the separation vessel

    if limin > len(lnorms_end):
        limin_final = limin - len(lnorms_end)
    else:
        limin_final = limin

    points_LICA = points_LICAMCA[:limin_final]
    points_LMCA = points_LICAMCA[limin_final:]

    # Definition of the indices to truncate (the equivalent of a radius of laca on each side)

    if limin <= len(lnorms_end):
        indice_LICA = geom.find_number_of_steps(
            points_LICA, center_LACA.get("center1")[1]
        )
        indice_LMCA = geom.find_number_of_steps(
            points_LMCA, center_LACA.get("center1")[1]
        )
    else:
        indice_LICA = geom.find_number_of_steps(
            points_LICA, center_LACA.get("center2")[1]
        )
        indice_LMCA = geom.find_number_of_steps(
            points_LMCA, center_LACA.get("center2")[1]
        )

    points_LICA = points_LICA[: points_LICA.shape[0] - indice_LICA]
    points_LMCA = points_LMCA[indice_LMCA:]

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
        rimin_final = rimin - len(rnorms_end)
    else:
        rimin_final = rimin

    points_RICA = points_RICAMCA[:rimin_final]
    points_RMCA = points_RICAMCA[rimin_final:]

    if rimin <= len(rnorms_end):
        indice_RICA = geom.find_number_of_steps(
            points_RICA, center_RACA.get("center1")[1]
        )
        indice_RMCA = geom.find_number_of_steps(
            points_RMCA, center_RACA.get("center1")[1]
        )
    else:
        indice_RICA = geom.find_number_of_steps(
            points_RICA, center_RACA.get("center2")[1]
        )
        indice_RMCA = geom.find_number_of_steps(
            points_RMCA, center_RACA.get("center2")[1]
        )

    points_RICA = points_RICA[: points_RICA.shape[0] - indice_RICA]
    points_RMCA = points_RMCA[indice_RMCA:]

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

    ax.view_init(30, 30)
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


def division_PCA(pinfo, case, step):
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

    pathctgr = (
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
    os.chdir(pathctgr)

    filesctgr = []
    for file in glob.glob("*.ctgr"):
        filesctgr.append(file)
    for files in filesctgr:
        if "BAS_PCA" in files:
            center_BAS = geom.get_center_radius_ulti(files, pinfo, case)
            side_bas = files[0]
            if side_bas == "L":
                other_side = "R"
            else:
                other_side = "L"
            for subfiles in filesctgr:
                if side_bas + "_Pcom" in subfiles:
                    center_bas_pcom = geom.get_center_radius_ulti(subfiles, pinfo, case)
                if other_side + "_Pcom" in subfiles:
                    center_non_bas_pcom = geom.get_center_radius_ulti(
                        subfiles, pinfo, case
                    )

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
            points_bas_pca = geom.get_spline_points(files, step)
            side_bas = files[0]
            if side_bas == "L":
                other_side = "R"
            else:
                other_side = "L"

            for subfile in onlyfiles:

                if other_side + "_PCA" in subfile:
                    points_non_bas_pca = geom.get_spline_points(subfile, step)

            # NEW METHOD - Works whatever the direction of the vessel

            target = [
                points_non_bas_pca[0],
                points_non_bas_pca[points_non_bas_pca.shape[0] - 1],
            ]
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
                limin_final = limin - len(lnorms_end)
            else:
                limin_final = limin

            # DIVISION BAS & PCA

            points_pca = points_bas_pca[limin_final:]
            points_bas = points_bas_pca[:limin_final]

            if limin <= len(lnorms_end):
                indice_bas = geom.find_number_of_steps(
                    points_pca, center_BAS.get("center1")[1]
                )
                indice_non_bas = geom.find_number_of_steps(
                    points_non_bas_pca, center_BAS.get("center1")[1]
                )
            else:
                indice_bas = geom.find_number_of_steps(
                    points_pca, center_BAS.get("center2")[1]
                )
                indice_non_bas = geom.find_number_of_steps(
                    points_non_bas_pca, center_BAS.get("center2")[1]
                )

            points_pca = points_pca[indice_bas:]
            points_non_bas_pca = points_non_bas_pca[indice_non_bas:]

            if side_bas == "L":
                side_vessel = "R"
            else:
                side_vessel = "L"

            for subfile in onlyfiles:

                if side_vessel + "_Pcom" in subfile:
                    points_Pcom = geom.get_spline_points(subfile, step)
                if side_bas + "_Pcom" in subfile:
                    points_bas_Pcom = geom.get_spline_points(subfile, step)

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

            if limin > len(bas_norms_end):
                limin_final = limin - len(bas_norms_end)
            else:
                limin_final = limin

            points_bas_P1 = points_pca[:limin_final]
            points_bas_P2 = points_pca[limin_final:]

            # Find the number of points to remove

            if limin <= len(lnorms_end):
                indice_bas_P1 = geom.find_number_of_steps(
                    points_bas_P1, center_bas_pcom.get("center1")[1]
                )
                indice_bas_P2 = geom.find_number_of_steps(
                    points_bas_P2, center_bas_pcom.get("center1")[1]
                )
            else:
                indice_bas_P1 = geom.find_number_of_steps(
                    points_bas_P1, center_bas_pcom.get("center2")[1]
                )
                indice_bas_P2 = geom.find_number_of_steps(
                    points_bas_P2, center_bas_pcom.get("center2")[1]
                )

            points_bas_P1 = points_bas_P1[: points_bas_P1.shape[0] - indice_bas_P1]
            points_bas_P2 = points_bas_P2[indice_bas_P2:]

            # SEPARATION P1 P2 NOT ON THE BASILAR SIDE

            nb_norms_start = []
            nb_norms_end = []
            for i in range(points_non_bas_pca.shape[0]):
                norm_start = np.linalg.norm(target[0] - points_non_bas_pca[i])
                norm_end = np.linalg.norm(target[1] - points_non_bas_pca[i])

                nb_norms_start.append(norm_start)
                nb_norms_end.append(norm_end)

            Ltot_norms = nb_norms_end + nb_norms_start
            lmini = np.min(Ltot_norms)
            limin = Ltot_norms.index(lmini)

            if limin > len(nb_norms_end):
                limin_final = limin - len(nb_norms_end)
            else:
                limin_final = limin

            points_non_bas_P1 = points_non_bas_pca[:limin_final]
            points_non_bas_P2 = points_non_bas_pca[limin_final:]

            # Find the number of points to delete to remove the radius of the non basilar side pcom

            if limin <= len(lnorms_end):
                indice_non_bas_P1 = geom.find_number_of_steps(
                    points_non_bas_P1, center_non_bas_pcom.get("center1")[1]
                )
                indice_non_bas_P2 = geom.find_number_of_steps(
                    points_non_bas_P2, center_non_bas_pcom.get("center1")[1]
                )
            else:
                indice_non_bas_P1 = geom.find_number_of_steps(
                    points_non_bas_P1, center_non_bas_pcom.get("center2")[1]
                )
                indice_non_bas_P2 = geom.find_number_of_steps(
                    points_non_bas_P2, center_non_bas_pcom.get("center2")[1]
                )

            points_non_bas_P1 = points_non_bas_P1[
                : points_non_bas_P1.shape[0] - indice_non_bas_P1
            ]
            points_non_bas_P2 = points_non_bas_P2[indice_non_bas_P2:]

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
                points_non_bas_P1[:, 0],
                points_non_bas_P1[:, 1],
                points_non_bas_P1[:, 2],
                label="Non Basilar side P1",
            )
            ax.scatter(
                points_non_bas_P2[:, 0],
                points_non_bas_P2[:, 1],
                points_non_bas_P2[:, 2],
                label="Non Basilar side P2",
            )

            ax.legend()
            ax.view_init(30, 90)
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
            if points_non_bas_P1.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = (
                    side_vessel + "_P1",
                    points_non_bas_P1,
                )
                k += 1
            if points_non_bas_P2.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = (
                    side_vessel + "_P2",
                    points_non_bas_P2,
                )
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
            points_laca = geom.get_spline_points(files, step)
        if "R_ACA" in files:
            points_raca = geom.get_spline_points(files, step)

    # VISUALIZATION

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(points_laca[:, 0], points_laca[:, 1], points_laca[:, 2], label="L_ACA")
    ax.scatter(points_raca[:, 0], points_raca[:, 1], points_raca[:, 2], label="R_ACA")
    ax.legend()
    ax.view_init(30, 60)

    # DIVISION L_ACA INTO LACA A1 & LACA A2

    step_point = int((len(points_laca)) / (len(points_laca) / 2))
    X, Y, Z = points_laca[:, 0], points_laca[:, 1], points_laca[:, 2]
    L = np.linspace(0, len(points_laca) - 1, 20)
    Lind = [int(np.floor(x)) for x in L]
    print(Lind)
    for ik in Lind:

        annotation = "point {}".format(ik)
        x, y, z = list(zip(X, Y, Z))[ik]
        ax.text(x, y, z, annotation)

    plt.show()

    print("\n")
    print("## Select separation point ##   " + "L_ACA" + "\n")
    for i in range(len(points_laca)):
        print("   ", i, " : point ", i)

    target = int(input("-->  "))

    points_LA1 = points_laca[:target]
    points_LA2 = points_laca[target:]

    # DIVISION R_ACA INTO RACA A1 & RACA A2

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(points_laca[:, 0], points_laca[:, 1], points_laca[:, 2], label="L_ACA")
    ax.scatter(points_raca[:, 0], points_raca[:, 1], points_raca[:, 2], label="R_ACA")
    ax.legend()
    ax.view_init(30, 60)

    step_point = int((len(points_raca)) / (len(points_raca) / 2))
    X, Y, Z = points_raca[:, 0], points_raca[:, 1], points_raca[:, 2]
    L = np.linspace(0, len(points_raca) - 1, 20)
    Lind = [int(np.floor(x)) for x in L]
    print(Lind)
    for ik in Lind:

        annotation = "point {}".format(ik)
        x, y, z = list(zip(X, Y, Z))[ik]
        ax.text(x, y, z, annotation)

    plt.show()

    print("\n")
    print("## Select separation point ##   " + "L_ACA" + "\n")
    for i in range(len(points_raca)):
        print("   ", i, " : point ", i)

    target = int(input("-->  "))

    points_RA1 = points_raca[:target]
    points_RA2 = points_raca[target:]

    # Final Visualization

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(
        points_LA1[:, 0], points_LA1[:, 1], points_LA1[:, 2], c="g", label="LACA A1"
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

    ax.view_init(30, 60)
    ax.legend()

    dpoints_divided = {}
    k = 0
    if points_LA1.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = "LA1", points_LA1
        k += 1
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

            I_supp.append(j)

        if (
            dpoint_i.get("points{}".format(j))[0] is not None
            and "L_ACA" in dpoint_i.get("points{}".format(j))[0]
        ):

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
        dvectors["vectors{}".format(j)] = filename, geom.calculate_normal_vectors(
            points
        )
        j += 1
    return dpoints, dvectors


# %% Main


def _main_(pinfo, case, step):

    dpoint_i = geom.create_dpoint(pinfo, case, step)

    # Step 2# Divide

    dpoints_divI = division_ICA(pinfo, case, step)
    dpoints_divACA = division_ACAs(pinfo, case, step)
    dpoints_divPCA = division_PCA(pinfo, case, step)

    dpoints = dpoint_i.copy()

    # Step 3# Add

    dpoints = add_divided_arteries(dpoints, dpoints_divI)
    dpoints = add_divided_arteries(dpoints, dpoints_divACA)
    dpoints = add_divided_arteries(dpoints, dpoints_divPCA)

    # Delete

    dpoints, indices = delete_old_arteries(dpoints)

    dpoints, dvectors = createfinal_dicts(dpoints, indices)

    return dpoints, dvectors
