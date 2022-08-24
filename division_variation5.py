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
       
            
            
    if points_LACA.shape[0]<points_RACA.shape[0]:
        missing_side='L'
        nmssg='R'
        points_m_ACA=points_LACA
        points_nm_ACA = points_RACA
    else:
        missing_side = 'R'
        nmssg='L'
        points_m_ACA = points_RACA
        points_nm_ACA = points_LACA
        
    for files in onlyfiles:
        if missing_side + "_ICA_MCA" in files:
            points_m_ICAMCA = geom.space_array(files,length)
        if nmssg + "_ICA_MCA" in files:
            points_nm_ICAMCA = geom.space_array(files,length)
            
    
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
        if nmssg + "_ACA" in files:
            print(files)
            center_nm_ACA = geom.get_center_radius_ulti(files, pinfo, case)

    # Manual division on the missing side | No Left ACA A1 to find the
    # intersection

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid()
    ax.scatter(
        points_m_ICAMCA[:, 0],
        points_m_ICAMCA[:, 1],
        points_m_ICAMCA[:, 2],
        label=missing_side + " ICA",
    )
    ax.scatter(
        points_nm_ICAMCA[:, 0],
        points_nm_ICAMCA[:, 1],
        points_nm_ICAMCA[:, 2],
        label=nmssg + " ICA ",
    )
    ax.scatter(
        points_m_ACA[:, 0], points_m_ACA[:, 1], points_m_ACA[:, 2], label=missing_side+" ACA "
    )
    ax.scatter(
        points_nm_ACA[:, 0], points_nm_ACA[:, 1], points_nm_ACA[:, 2], label=nmssg + " ACA "
    )
    ax.view_init(0, 90)
    ax.legend()

    step_point = int((len(points_m_ICAMCA)) / (len(points_m_ICAMCA) / 2))
    X, Y, Z = points_m_ICAMCA[:, 0], points_m_ICAMCA[:, 1], points_m_ICAMCA[:, 2]
    L = np.linspace(0, len(points_m_ICAMCA) - 1, 20)
    Lind = [int(np.floor(x)) for x in L]
    print(Lind)
    for ik in Lind:
        annotation = "point {}".format(ik)
        x, y, z = list(zip(X, Y, Z))[ik]
        ax.text(x, y, z, annotation)

    plt.show()

    print("\n")
    print("## Select separation point ##   " + "LICAMCA" + "\n")
    for i in range(len(points_m_ICAMCA)):
        print("   ", i, " : point ", i)

    target = int(input("-->  "))

    points_m_ICA = points_m_ICAMCA[:target]
    points_m_MCA = points_m_ICAMCA[target:]

    # Automatic method for the Right side | Finding the minimum of norms to
    # find the intersection with ACA
    
    points_nm_ICA,points_nm_MCA=geom.bifurcation_and_radius_remove(points_nm_ICAMCA, points_nm_ACA, center_nm_ACA)

    #points_m_ICA,points_nm_ICA=geom.crop_ICAs(pinfo, case, points_nm_ICA, points_nm_ICA)
    
    
    # rtarget = [points_RACA[0], points_RACA[points_nm_ACA.shape[0] - 1]]
    # rnorms_end = []
    # rnorms_start = []
    # for i in range(points_RICAMCA.shape[0]):
    #     norm_end = np.linalg.norm(rtarget[1] - points_RICAMCA[i])
    #     norm_start = np.linalg.norm(rtarget[0] - points_RICAMCA[i])
    #     rnorms_end.append(norm_end)
    #     rnorms_start.append(norm_start)

    # Ltot_norms = rnorms_end + rnorms_start
    # rmini = np.min(Ltot_norms)
    # rimin = Ltot_norms.index(rmini)
    # if rimin > len(rnorms_end):
    #     rimin -= len(rnorms_end)

    # points_RICA = points_RICAMCA[:rimin]
    # points_RMCA = points_RICAMCA[rimin:]

    # FINAL VISUALIZATION

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid()

    ax.scatter(
        points_m_ICA[:, 0], points_m_ICA[:, 1], points_m_ICA[:, 2], label=missing_side + " ICA"
    )
    ax.scatter(
        points_nm_ICA[:, 0], points_nm_ICA[:, 1], points_nm_ICA[:, 2], label=nmssg + " ICA "
    )
    ax.scatter(
        points_m_MCA[:, 0], points_m_MCA[:, 1], points_m_MCA[:, 2], label=missing_side + " MCA"
    )
    ax.scatter(
        points_nm_MCA[:, 0], points_nm_MCA[:, 1], points_nm_MCA[:, 2], label=nmssg + " MCA"
    )
    ax.scatter(
        points_m_ACA[:, 0], points_m_ACA[:, 1], points_m_ACA[:, 2], label=missing_side + " ACA "
    )
    ax.scatter(
        points_nm_ACA[:, 0], points_nm_ACA[:, 1], points_nm_ACA[:, 2], label=nmssg + " ACA "
    )

    #ax.view_init(30, 90)
    ax.legend()
    plt.show()

    dpoints_divided = {}
    k = 0
    if points_m_ICA.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = missing_side + "_ICA", points_m_ICA
        k += 1
    if points_nm_ICA.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = nmssg + "_ICA", points_nm_ICA
        k += 1
    if points_m_MCA.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = missing_side + "_MCA", points_m_MCA
        k += 1
    if points_nm_MCA.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = nmssg+"_MCA", points_nm_MCA
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
            points_LACA = geom.space_array(files,length)
        if "R_ACA" in files:
            points_RACA = geom.space_array(files,length)
       
            
            
    if points_LACA.shape[0]<points_RACA.shape[0]:
        missing_side='L'
        nmssg='R'
        points_m_ACA=points_LACA
        points_nm_ACA = points_RACA
    else:
        missing_side = 'R'
        nmssg='L'
        points_m_ACA = points_RACA
        points_nm_ACA = points_LACA
        
    for files in onlyfiles:
        if nmssg + "_ACA" in files:
            points_nm_ICA_MCA = geom.space_array(files,length)

 
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
        if missing_side + "_ACA" in files:
            print(files)
            center_m_ACA = geom.get_center_radius_ulti(files, pinfo, case)

    #DIVISION Regular side : A1 & A2
         
    L_dir_Ls=np.min([np.linalg.norm(points_nm_ACA[0]-x) for x in points_nm_ICA_MCA])
    L_dir_Le=np.min([np.linalg.norm(points_nm_ACA[points_nm_ACA.shape[0]-1]-x) for x in points_nm_ICA_MCA])
          
    if L_dir_Le < L_dir_Ls:
        print("ACA inverted")
        points_nm_ACA=points_nm_ACA[::-1]
    
    
    points_nm_A1,points_nm_A2 = geom.bifurcation_and_radius_remove(points_nm_ACA, points_m_ACA, center_m_ACA)
    
    # DIVISION Acom & A1 : manual

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(
        points_nm_A2[:, 0],
        points_nm_A2[:, 1],
        points_nm_A2[:, 2],
        c="k",
        label=nmssg + " A2",
    )
    ax.scatter(
        points_nm_A1[:, 0],
        points_nm_A1[:, 1],
        points_nm_A1[:, 2],
        c="b",
        label=nmssg + " A1",
    )
    X, Y, Z = points_m_ACA[:, 0], points_m_ACA[:, 1], points_m_ACA[:, 2]
    ax.plot(X, Y, Z, "--")
    # ax.plot(X, Y, Z, "o", label=ves)

    ax.legend()
    plt.title('ACA Acom separation')
    # i_point=0
    step_point = int((len(points_m_ACA)) / (len(points_m_ACA) / 2))

    L = np.linspace(0, len(points_m_ACA) - 1, 20)
    Lind = [int(np.floor(x)) for x in L]
    print(Lind)
    for ik in Lind:

        annotation = "point {}".format(ik)
        x, y, z = list(zip(X, Y, Z))[ik]
        ax.text(x, y, z, annotation)
    ax.view_init(30,30)
    plt.show()

    print("\n")
    print("## Select separation point ##   " + missing_side + ' ACA' + "\n")
    for i in range(len(points_m_ACA)):
        print("   ", i, " : point ", i)

    target = int(input("-->  "))

    points_m_Acom = points_m_ACA[target:]
    points_m_A2 = points_m_ACA[:target]





    # Final Visualization

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.scatter(
        points_m_Acom[:, 0], points_m_Acom[:, 1], points_m_Acom[:, 2], c="g", label="Acom"
    )
    
    ax.scatter(
        points_nm_A2[:, 0], points_nm_A2[:, 1], points_nm_A2[:, 2], c="b", label=nmssg + " ACA A2"
    )
    ax.scatter(
        points_nm_A1[:, 0], points_nm_A1[:, 1], points_nm_A1[:, 2], c="k", label=nmssg + " ACA A1"
    )
    ax.scatter(
        points_m_A2[:, 0], points_m_A2[:, 1], points_m_A2[:, 2], c="b", label=missing_side + " ACA A2"
    )

    ax.view_init(30, 30)
    ax.legend()

    dpoints_divided = {}
    k = 0
    # if points_LA1.shape[0] != 0:
    #     dpoints_divided["points{}".format(k)] = "LA1", points_LA1
    #     k += 1
    if points_nm_A2.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = nmssg + "A2", points_nm_A2
        k += 1
    if points_nm_A1.shape[0] != 0:
        dpoints_divided["points{}".format(k)] = nmssg + "A1", points_nm_A1
        k += 1
    if points_m_A2.shape[0] != 0:
        dpoints_divided["points{}".format(k)] =missing_side + "A2", points_m_A2
        k += 1

    return dpoints_divided


def division_PCA(pinfo, case, length):
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
                if other_side + "_Pcom_PCA" in subfiles:
                    center_non_bas_P1 = geom.get_center_radius_ulti(
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

            print(other_side)
            for subfile in onlyfiles:
                if other_side + '_PCA_P1' in subfile:
                    points_non_bas_P1=geom.space_array(subfile,length)
                    division_case='nb_divided'
                    for subsubfile in onlyfiles:
                        if other_side + '_Pcom_PCA' in subsubfile:
                            points_non_bas_pcompca=geom.space_array(subsubfile,length)
                if (other_side + "_PCA") in subfile:
                    if len(subfile)==9:
                        points_non_bas_pca = geom.space_array(subfile,length)
                        division_case='regular'
                    

            for subfile in onlyfiles:
                
                
                if side_bas + "_Pcom" in subfile:
                    points_bas_Pcom = geom.space_array(subfile,length)
                    
                if division_case=='regular':
                    if other_side + "_Pcom" in subfile:
                        
                        points_Pcom = geom.space_array(subfile,length)
            print(points_bas_Pcom)
            
            if division_case=='regular':
                
                
                # Step 1 : divide pca & bas
                
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
            
                # Divide P1 & P2
                
                points_bas_P1,points_bas_P2=geom.bifurcation_and_radius_remove(points_pca, points_bas_Pcom, center_bas_pcom)
                points_non_bas_P1,points_non_bas_P2=geom.bifurcation_and_radius_remove(points_non_bas_pca, points_Pcom, center_non_bas_pcom)
              
                #Attention peut-être mal tronqué pca
                    
            if division_case == 'nb_divided':
                points_bas,points_pca,case_center = geom.bifurcation(points_bas_pca, points_non_bas_P1)
                
                L_dir_Ls=np.min([np.linalg.norm(points_pca[0]-x) for x in points_bas])
                L_dir_Le=np.min([np.linalg.norm(points_pca[points_pca.shape[0]-1]-x) for x in points_bas])
                if L_dir_Le < L_dir_Ls:
                    print("basilar side pca inverted")
                    points_pca=points_pca[::-1]
                    
                  
                points_pca,points_non_bas_P1=geom.remove_center(points_pca, points_non_bas_P1, center_BAS, case_center)

                points_non_bas_P2,points_non_bas_Pcom=geom.bifurcation_and_radius_remove(points_non_bas_pcompca, points_non_bas_P1, center_non_bas_P1)
                
                points_bas_P1,points_bas_P2=geom.bifurcation_and_radius_remove(points_pca, points_bas_Pcom, center_bas_pcom)
               

          

           
            #Plot final visualization
            
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
            if division_case=='regular':
                ax.scatter(
                    points_Pcom[:, 0],
                    points_Pcom[:, 1],
                    points_Pcom[:, 2],
                    label="non basilar side Pcom",
            )
            if division_case=='nb_divided':
                ax.scatter(
                    points_non_bas_Pcom[:, 0],
                    points_non_bas_Pcom[:, 1],
                    points_non_bas_Pcom[:, 2],
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
                    other_side + "_P1",
                    points_non_bas_P1,
                )
                k += 1
            if points_non_bas_P2.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = (
                    other_side + "_P2",
                    points_non_bas_P2,
                )
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
        dvectors["vectors{}".format(j)] = filename, geom.calculate_normal_vectors(points)
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
