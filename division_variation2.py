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


def division_ICA(pinfo, case, length):
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
            points_LACA = geom.space_array(files,length)
        if "R_ACA" in files:
            points_RACA = geom.space_array(files,length)
        if "L_ICA_MCA" in files:
            points_LICAMCA = geom.space_array(files,length)
        if "R_ICA_MCA" in files:
            points_RICAMCA = geom.space_array(files,length)

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


    
    points_LICA,points_LMCA=geom.bifurcation_and_radius_remove(points_LICAMCA, points_LACA, center_LACA)
    points_RICA,points_RMCA=geom.bifurcation_and_radius_remove(points_RICAMCA, points_RACA, center_RACA)

    points_LICA,points_RICA=geom.crop_ICAs(pinfo,case,points_LICA,points_RICA)

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


def division_PCA(pinfo, case,length):
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
            points_bas_pca = geom.space_array(files,length)
            side_bas = files[0]
            if side_bas == "L":
                other_side = "R"
            else:
                other_side = "L"

            for subfile in onlyfiles:

                if other_side + "_PCA" in subfile:
                    points_non_bas_pca = geom.space_array(subfile,length)

         
            
            points_bas,points_pca,case_center = geom.bifurcation(points_bas_pca, points_non_bas_pca)
            
            # Step 2 : organize in an other direction if necessary
            
            L_dir_Ls=np.min([np.linalg.norm(points_pca[0]-x) for x in points_bas])
            L_dir_Le=np.min([np.linalg.norm(points_pca[points_pca.shape[0]-1]-x) for x in points_bas])
            if L_dir_Le < L_dir_Ls:
                print("basilar side pca inverted")
                points_pca=points_pca[::-1]
          
                # Non basilar side
            L_dir_Rs=np.min([np.linalg.norm(points_non_bas_pca[0]-x) for x in points_bas])
            L_dir_Re=np.min([np.linalg.norm(points_non_bas_pca[points_non_bas_pca.shape[0]-1]-x) for x in points_bas])
            if L_dir_Re < L_dir_Rs:
                print("non basilar side pca inverted")
                points_non_bas_pca=points_non_bas_pca[::-1]
                
            # Step 3 : remove a radius of the bifurcation vessel
            
            points_pca,points_non_bas_pca=geom.remove_center(points_pca, points_non_bas_pca, center_BAS, case_center)
            

            if side_bas == "L":
                side_vessel = "R"
            else:
                side_vessel = "L"

            for subfile in onlyfiles:

                if side_vessel + "_Pcom" in subfile:
                    points_Pcom = geom.space_array(subfile,length)
                if side_bas + "_Pcom" in subfile:
                    points_bas_Pcom = geom.space_array(subfile,length)

                # elif side_vessel + '_PCA' in subfile:
                #     points_pca=get_spline_points(subfile,step)


            points_bas_P1,points_bas_P2=geom.bifurcation_and_radius_remove(points_pca, points_bas_Pcom, center_bas_pcom)
            points_non_bas_P1,points_non_bas_P2 = geom.bifurcation_and_radius_remove(points_non_bas_pca, points_Pcom, center_non_bas_pcom)
            
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


def division_ACAs(pinfo, case, length):
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
            points_laca = geom.space_array(files,length)
        if "R_ACA" in files:
            points_raca = geom.space_array(files,length)
        
    for files in onlyfiles:
        if 'L_ICA_MCA' in files:
             points_LICA_MCA=geom.space_array(files,length)
        if 'R_ICA_MCA' in files:
             points_RICA_MCA=geom.space_array(files,length)
         

    # VISUALIZATION

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(points_laca[:, 0], points_laca[:, 1], points_laca[:, 2], label="L_ACA")
    ax.scatter(points_raca[:, 0], points_raca[:, 1], points_raca[:, 2], label="R_ACA")
    ax.legend()
    ax.view_init(30, 60)
    
    # PLACE THE ACAs IN THE GOOD ORDER
    
    L_dir_Ls=np.min([np.linalg.norm(points_laca[0]-x) for x in points_LICA_MCA])
    L_dir_Le=np.min([np.linalg.norm(points_laca[points_laca.shape[0]-1]-x) for x in points_LICA_MCA])
      
    if L_dir_Le < L_dir_Ls:
        print("ACA inverted")
        points_laca=points_laca[::-1]
       
    L_dir_Rs=np.min([np.linalg.norm(points_raca[0]-x) for x in points_RICA_MCA])
    L_dir_Re=np.min([np.linalg.norm(points_raca[points_raca.shape[0]-1]-x) for x in points_RICA_MCA])

    if L_dir_Re < L_dir_Rs:
        print("right ACA inverted")
        points_raca=points_raca[::-1]

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


def _main_(pinfo, case, length):

    dpoint_i = geom.create_dpoint(pinfo, case, length)

    # Step 2# Divide

    dpoints_divI = division_ICA(pinfo, case, length)
    dpoints_divACA = division_ACAs(pinfo, case, length)
    dpoints_divPCA = division_PCA(pinfo, case, length)

    dpoints = dpoint_i.copy()


    # Step 3# Add

    dpoints = add_divided_arteries(dpoints, dpoints_divI)
    dpoints = add_divided_arteries(dpoints, dpoints_divACA)
    dpoints = add_divided_arteries(dpoints, dpoints_divPCA)

    # Delete

    dpoints, indices = delete_old_arteries(dpoints)

    dpoints, dvectors = createfinal_dicts(dpoints, indices)

    return dpoints, dvectors
