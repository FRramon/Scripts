# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:16:42 2022

@author: GALADRIEL_GUEST
"""


# Script whose main function returns the actual divided geometry for the first variation
# --> no right P1

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

#%% Import intern modules

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
            points_LACA = geom.space_array(files, length)
        if "R_ACA" in files:
            points_RACA = geom.space_array(files, length)
        if "L_ICA_MCA" in files:
            points_LICAMCA = geom.space_array(files, length)
        if "R_ICA_MCA" in files:
            points_RICAMCA = geom.space_array(files, length)

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

    points_LICA,points_LMCA=geom.bifurcation_and_radius_remove(points_LICAMCA,points_LACA,center_LACA)
    points_RICA,points_RMCA=geom.bifurcation_and_radius_remove(points_RICAMCA,points_RACA,center_RACA)
      
    points_LICA,points_RICA=geom.crop_ICAs(pinfo, case, points_LICA, points_RICA)
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


def division_RP(pinfo, case, length):
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

    vessel = "R_Pcom_PCA"

    if vessel[0] == "L":
        other_side = "R"
    else:
        other_side = "L"

    for file in glob.glob("*.pth"):
        onlyfiles.append(file)
    for files in onlyfiles:
        if "Pcom_PCA" in files:
            points_vessel = geom.space_array(files, length)
        if "BAS_PCA" in files:
            points_bas = geom.space_array(files, length)
        if other_side + "_Pcom" in files:
            points_pcom = geom.space_array(files, length)

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
        if other_side + "_Pcom" in files:
            center_pcom = geom.get_center_radius_ulti(files, pinfo, case)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(
        points_bas[:, 0],
        points_bas[:, 1],
        points_bas[:, 2],
        c="k",
        label=other_side + "_BAS_PCA",
    )
    ax.scatter(
        points_pcom[:, 0],
        points_pcom[:, 1],
        points_pcom[:, 2],
        c="b",
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
    # print(Lind)
    for ik in Lind:

        annotation = "point {}".format(ik)
        x, y, z = list(zip(X, Y, Z))[ik]
        ax.text(x, y, z, annotation)

    plt.show()

    print("\n")
    print("## Select separation point ##   " + vessel[:-4] + "\n")
    for i in range(len(points_vessel)):
        print("   ", i, " : point ", i)

    target = int(input("-->  "))

    points_1 = points_vessel[target:]
    points_2 = points_vessel[:target]

    indice_p2 = geom.find_number_of_steps(points_2, center_pcom.get("center2")[1])
    points_1 = points_1[indice_p2:]

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
        points_1[:, 0], points_1[:, 1], points_1[:, 2], label=vessel[0] + "PCA P2"
    )
    ax.scatter(
        points_2[:, 0], points_2[:, 1], points_2[:, 2], label=vessel[0] + "_Pcom"
    )

    ax.legend()

    dpoints_divided = {}
    k = 0
    if points_1.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = vessel[0] + "P2", points_1
        k += 1
    if points_2.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = vessel[0] + "_Pcom", points_2
        k += 1

    return dpoints_divided


def division_A(pinfo, case, length):
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

    # folder = "_segmentation_cropACA"
    # pathpath = (
    #     "N:/vasospasm/"
    #     + pinfo
    #     + "/"
    #     + case
    #     + "/1-geometry/"
    #     + pinfo
    #     + "_"
    #     + case
    #     + folder
    #     + "/Paths"
    # )

    # os.chdir(pathpath)
    # print(pathpath)
    # onlyfiles = []
    # for file in glob.glob("*.pth"):
    #     onlyfiles.append(file)
    # for files in onlyfiles:
    #     if "Acom" in files:
    #         points_Acom = geom.get_spline_points(files, step)
    #     if "L_ACA" in files:
    #         points_LACA = geom.get_spline_points(files, step)
    #     if "R_ACA" in files:
    #         points_RACA = geom.get_spline_points(files, step)
            

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
        if "Acom" in files:
            center_Acom = geom.get_center_radius_ulti(files, pinfo, case)
        if "Acom_posterior" in files:
            center_Acom_post = geom.get_center_radius_ulti(files, pinfo, case)
        if 'ACA_A2' in files:
            center_A2 = geom.get_center_radius_ulti(files, pinfo, case)

 
    folder = "_segmentation" # HERE ADD '_cropACA' if issue with path&model size
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
        if "_ACA_A1" in files:
            points_ACA_Acom = geom.space_array(files, length)
            division_case='divided_acom'
            side_change=files[0]
            if side_change=='L':
                other_side='R'
            else:
                other_side='L'
            # print(other_side)
            for subfile in onlyfiles:
                if 'ACA_A2' in subfile:
                    points_ACA_A2 = geom.space_array(subfile, length)
                    
                if other_side + '_ACA' in subfile:
                    points_other_ACA = geom.space_array(subfile, length)
                if 'Acom_posterior' in subfile:
                    points_Acom_post=geom.space_array(subfile, length)
                
        if 'L_ACA' in files:
            if len(files)==9:
                division_case='regular'
                for subfiles in onlyfiles:
                    if "Acom" in subfiles:
                        points_Acom = geom.space_array(subfiles, length)
                    if "L_ACA" in subfiles:
                        points_LACA = geom.space_array(subfiles, length)
                    if "R_ACA" in subfiles:
                        points_RACA = geom.space_array(subfiles, length)
    
    for files in onlyfiles:
        if 'L_ICA_MCA' in files:
            points_LICA_MCA=geom.space_array(files, length)
        if 'R_ICA_MCA' in files:
            points_RICA_MCA=geom.space_array(files, length)
        
    print(division_case)

    if division_case=='divided_acom':
        
    # STEP 1 : points ACA
        #Left side
        L_dir_Ls=np.min([np.linalg.norm(points_ACA_Acom[0]-x) for x in points_LICA_MCA])
        L_dir_Le=np.min([np.linalg.norm(points_ACA_Acom[points_ACA_Acom.shape[0]-1]-x) for x in points_LICA_MCA])
           
        if L_dir_Le < L_dir_Ls:
            print("ACA A1 inverted")
            points_ACA_Acom=points_ACA_Acom[::-1]
        
        L_dir_Ls2=np.min([np.linalg.norm(points_ACA_A2[0]-x) for x in points_ACA_Acom])
        L_dir_Le2=np.min([np.linalg.norm(points_ACA_A2[points_ACA_A2.shape[0]-1]-x) for x in points_ACA_Acom])
           
        
        if L_dir_Le2 < L_dir_Ls2:
            print("ACA A2 inverted")
            points_ACA_A2=points_ACA_A2[::-1]
        
        # Right side
        L_dir_Rs=np.min([np.linalg.norm(points_other_ACA[0]-x) for x in points_RICA_MCA])
        L_dir_Re=np.min([np.linalg.norm(points_other_ACA[points_other_ACA.shape[0]-1]-x) for x in points_RICA_MCA])
    
        if L_dir_Re < L_dir_Rs:
            print("right ACA inverted")
            points_RACA=points_RACA[::-1]
            
      
             
        points_LA1=points_ACA_Acom
        points_LA2=points_ACA_A2
        
        points_RA1,points_RA2=geom.bifurcation_and_radius_remove(points_other_ACA, points_Acom_post,center_Acom_post)
        
        
       
    elif division_case=='regular':
        
        
        L_dir_Ls=np.min([np.linalg.norm(points_LACA[0]-x) for x in points_LICA_MCA])
        L_dir_Le=np.min([np.linalg.norm(points_LACA[points_LACA.shape[0]-1]-x) for x in points_LICA_MCA])
          
        if L_dir_Le < L_dir_Ls:
           print("ACA inverted")
           points_LACA=points_LACA[::-1]
           
        L_dir_Rs=np.min([np.linalg.norm(points_RACA[0]-x) for x in points_RICA_MCA])
        L_dir_Re=np.min([np.linalg.norm(points_RACA[points_RACA.shape[0]-1]-x) for x in points_RICA_MCA])
    
        if L_dir_Re < L_dir_Rs:
            print("right ACA inverted")
            points_RACA=points_RACA[::-1]
          
        points_LA1,points_LA2=geom.bifurcation_and_radius_remove(points_LACA, points_Acom, center_Acom)
          
        points_RA1,points_RA2=geom.bifurcation_and_radius_remove(points_RACA, points_Acom,center_Acom)
        

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid()
    if division_case=='divided_acom':
        
        ax.scatter(
            points_Acom_post[:, 0], points_Acom_post[:, 1], points_Acom_post[:, 2], c="k", label="Acom"
        )
    if division_case=='regular':
        ax.scatter(
            points_Acom[:, 0], points_Acom[:, 1], points_Acom[:, 2], c="k", label="Acom"
        )
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

    ax.view_init(30,60)
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


def new_division_P_bas(pinfo, case, length):
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
            for subfiles in filesctgr:
                if side_bas + "_Pcom" in subfiles:
                    center_pcom = geom.get_center_radius_ulti(subfiles, pinfo, case)

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
            points_bas_pca = geom.space_array(files, length)
            side_bas = files[0]

            for subfile in onlyfiles:

                if side_bas + "_Pcom" in subfile:
                    points_target = geom.space_array(subfile, length)

                if "Pcom_PCA" in subfile:
                    points_otherside = geom.space_array(subfile, length)

            
            points_basP1,points_P2=geom.bifurcation_and_radius_remove(points_bas_pca, points_target, center_pcom)

            # DIVISION BAS & P1

            # PLOT AND DIVIDE MANUALLY

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
            ax.scatter(X, Y, Z, "o", label="BAS + P1")

            ax.legend()
            plt.title("BAS + P1 Division")
            L = np.linspace(0, points_basP1.shape[0] - 1, 20)
            Lind = [int(np.floor(x)) for x in L]
            # print(Lind)
            for ik in Lind:

                annotation = "point {}".format(ik)
                x, y, z = list(zip(X, Y, Z))[ik]
                ax.text(x, y, z, annotation)
            plt.show()

            print("\n")
            print("## Select separation point ##   " + "BAS_P1" + "\n")
            for i in range(len(points_basP1)):
                print("   ", i, " : point ", i)

            target = int(input("-->  "))

            plt.show()
            points_bas = points_basP1[:target]
            points_P1 = points_basP1[target:]

            # REMOVE RADIUS OF BAS IN THE LEFT P1

            indice_bas = geom.find_number_of_steps(
                points_P1, center_BAS.get("center1")[1]
            )

            points_P1 = points_P1[indice_bas:]

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

            I_supp.append(j)

        if (
            dpoint_i.get("points{}".format(j))[0] is not None
            and "L_PCA" in dpoint_i.get("points{}".format(j))[0]
        ):

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
        
        if (
            dpoint_i.get("points{}".format(j))[0] is not None
            and "BAS_PCA" in dpoint_i.get("points{}".format(j))[0]
        ):
            I_supp.append(j)
        if (
            dpoint_i.get("points{}".format(j))[0] is not None
            and "Pcom_PCA" in dpoint_i.get("points{}".format(j))[0]
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

    print("patient info : ", pinfo + ' ' + case)
    dpoint_i = geom.create_dpoint(pinfo, case, length)

    # Step 2# CREATE NEW DIVIDED VESSELS

    dpoints_divI = division_ICA(pinfo, case, length)
    dpoints_divACA = division_A(pinfo, case, length)
    dpoints_divLPCA = new_division_P_bas(pinfo, case, length)
    dpoints_divRP = division_RP(pinfo, case, length)

    dpoints = dpoint_i.copy()

    # Step 3# DELETE THE OLD VESSELS

    dpoints = add_divided_arteries(dpoints, dpoints_divI)
    dpoints = add_divided_arteries(dpoints, dpoints_divACA)
    dpoints = add_divided_arteries(dpoints, dpoints_divLPCA)
    dpoints = add_divided_arteries(dpoints, dpoints_divRP)

    dpoints, indices = delete_old_arteries(dpoints)

    dpoints, dvectors = createfinal_dicts(dpoints, indices)

    return dpoints, dvectors
