# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:50:26 2022

@author: GALADRIEL_GUEST
"""

# Script whose main function returns the actual divided geometry for the seventh variation
# -->No left PCom

# %% Imports

import glob
import os as os

import numpy as np
import matplotlib.pyplot as plt

import importlib
import pickle

#%% Import intern geometric module

os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

import geometry_slice as geom

importlib.reload(geom)

#%% Functions


def division_ICA(pinfo, case, length):
    """


    Parameters
    ----------
    pinfo : str, example : 'pt2' , 'vsp7'
    case : str, 'baseline' or 'vasospasm'
    length : the length between each points

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

# -> An ACA start from the bifurcation w/ ICA

def division_ACA(pinfo, case, length):
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
        if "Acom" in files:
            center_Acom = geom.get_center_radius_ulti(files, pinfo, case)
        if "Acom_posterior" in files:
            center_Acom_post = geom.get_center_radius_ulti(files, pinfo, case)
        if 'ACA_A2' in files:
            center_A2 = geom.get_center_radius_ulti(files, pinfo, case)

        

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
        if "_ACA_A1" in files:
            points_ACA_Acom = geom.space_array(files,length)
            division_case='divided_acom'
            side_change=files[0]
            if side_change=='L':
                other_side='R'
            else:
                other_side='L'
            print(other_side)
            for subfile in onlyfiles:
                if 'ACA_A2' in subfile:
                    points_ACA_A2 = geom.space_array(subfile,length)
                    
                if other_side + '_ACA' in subfile:
                    points_other_ACA = geom.space_array(subfile,length)
                if 'Acom_posterior' in subfile:
                    points_Acom_post=geom.space_array(subfile,length)
                
        if 'L_ACA' in files:
            if len(files)==9:
                division_case='regular'
                for subfiles in onlyfiles:
                    if "Acom" in subfiles:
                        points_Acom = geom.space_array(subfiles,length)
                    if "L_ACA" in subfiles:
                        points_LACA = geom.space_array(subfiles,length)
                    if "R_ACA" in subfiles:
                        points_RACA = geom.space_array(subfiles,length)
    
    for files in onlyfiles:
        if 'L_ICA_MCA' in files:
            points_LICA_MCA=geom.space_array(files,length)
        if 'R_ICA_MCA' in files:
            points_RICA_MCA=geom.space_array(files,length)
        
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





# def division_A(pinfo, case, length):
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

#     # pathpath = "C:/Users/Francois/Desktop/Stage_UW/" + pinfo + "/path"

#     if pinfo == "pt2":
#         folder = "_segmentation_no_vti"
#     else:
#         folder = "_segmentation"
#     pathpath = (
#         "N:/vasospasm/"
#         + pinfo
#         + "/"
#         + case
#         + "/1-geometry/"
#         + pinfo
#         + "_"
#         + case
#         + folder
#         + "/paths"
#     )

#     os.chdir(pathpath)
#     onlyfiles = []
#     for file in glob.glob("*.pth"):
#         onlyfiles.append(file)
#     for files in onlyfiles:
#         if "Acom" in files:
#             points_Acom = geom.space_array(files,length)
#         if "L_ACA" in files:
#             points_LACA = geom.space_array(files,length)
#         if "R_ACA" in files:
#             points_RACA = geom.space_array(files,length)

#     # Assuming that Acom is well oriented from left to right

#     target = [points_Acom[0], points_Acom[points_Acom.shape[0] - 1]]

#     Lnorms_start = []
#     Lnorms_end = []
#     for i in range(points_LACA.shape[0]):
#         lnorm_start = np.linalg.norm(target[0] - points_LACA[i])
#         lnorm_end = np.linalg.norm(target[1] - points_LACA[i])

#         Lnorms_start.append(lnorm_start)
#         Lnorms_end.append(lnorm_end)
#     Lnorms_tot = Lnorms_end + Lnorms_start

#     lmini = np.min(Lnorms_tot)
#     limin = Lnorms_tot.index(lmini)
#     if limin > len(Lnorms_end):
#         limin -= len(Lnorms_end)

#     points_LA1 = points_LACA[:limin]
#     points_LA2 = points_LACA[limin:]

#     Rnorms_start = []
#     Rnorms_end = []
#     for i in range(points_RACA.shape[0]):
#         lnorm_start = np.linalg.norm(target[0] - points_RACA[i])
#         lnorm_end = np.linalg.norm(target[1] - points_RACA[i])

#         Rnorms_start.append(lnorm_start)
#         Rnorms_end.append(lnorm_end)
#     Rnorms_tot = Lnorms_end + Lnorms_start

#     rmini = np.min(Rnorms_tot)
#     rimin = Rnorms_tot.index(rmini)
#     if rimin > len(Rnorms_end):
#         rimin -= len(Rnorms_end)

#     points_RA1 = points_RACA[:limin]
#     points_RA2 = points_RACA[limin:]

#     fig = plt.figure(figsize=(7, 7))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.grid()

#     ax.scatter(points_Acom[:, 0], points_Acom[:, 1], points_Acom[:, 2], label="Acom")
#     ax.scatter(
#         points_RA1[:, 0], points_RA1[:, 1], points_RA1[:, 2], label="RIGHT ACA A1"
#     )
#     ax.scatter(
#         points_RA2[:, 0], points_RA2[:, 1], points_RA2[:, 2], label="RIGHT ACA A2"
#     )
#     ax.scatter(
#         points_LA1[:, 0], points_LA1[:, 1], points_LA1[:, 2], label="LEFT ACA A1"
#     )
#     ax.scatter(
#         points_LA2[:, 0], points_LA2[:, 1], points_LA2[:, 2], label="LEFT ACA A2"
#     )

#     ax.view_init(30, 90)
#     ax.legend()
#     plt.show()

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


def division_P_bas(pinfo, case, length):
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
                    center_Pcom = geom.get_center_radius_ulti(subfiles, pinfo, case)
                if other_side + "_Pcom" in subfiles:
                    center_Pcom = geom.get_center_radius_ulti(
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
            print('yes')
            points_bas_pca = geom.space_array(files,length)
            side_bas = files[0]

            for subfile in onlyfiles:

                if side_bas == "L":
                    if "R_PCA" in subfile:
                        points_target = geom.space_array(subfile,length)

                else:
                    if "L_PCA" in subfile:
                        points_target = geom.space_array(subfile,length)

            
            points_bas,points_pca,case_center = geom.bifurcation(points_bas_pca, points_target)

            
            if side_bas == "L":
                side_vessel = "R"
            else:
                side_vessel = "L"

            for subfile in onlyfiles:
                if "_Pcom" in subfile:
                    side_pcom = subfile[0]
                    points_Pcom = geom.space_array(subfile,length)
                    
            print(side_pcom,side_bas)

            if side_pcom == side_bas:

                # Basilar side : Automatic

                L_dir_Ls=np.min([np.linalg.norm(points_pca[0]-x) for x in points_bas])
                L_dir_Le=np.min([np.linalg.norm(points_pca[points_pca.shape[0]-1]-x) for x in points_bas])
                if L_dir_Le < L_dir_Ls:
                    print("basilar side pca inverted")
                    points_pca=points_pca[::-1]
                
                points_bas_P1,points_bas_P2=geom.bifurcation_and_radius_remove(points_bas_pca, points_Pcom, center_Pcom)

                # Other side : Manual

                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="3d")
                ax.grid(False)
                
                points_vessel = points_target
                X, Y, Z = points_vessel[:, 0], points_vessel[:, 1], points_vessel[:, 2]
                ax.scatter(points_bas_pca[:,0],points_bas_pca[:,1],points_bas_pca[:,2],c='y',label='Basilar')
                ax.scatter(points_target[:,0],points_target[:,1],points_target[:,2])
                # ax.scatter(points_P1[:,0],points_P1[:,1],points_P1[:,2],c='k',label='PCA P1')
                # ax.scatter(points_P2[:,0],points_P2[:,1],points_P2[:,2],c='y',label='PCA P2')

                ax.plot(X, Y, Z, "--")
                ax.plot(X, Y, Z, "o")
                
                plt.legend()
                plt.title('pca sep')
                
                L = np.linspace(0, len(points_vessel) - 1, 20)
                Lind = [int(np.floor(x)) for x in L]
                print(Lind)
                for ik in Lind:
            
                    annotation = "point {}".format(ik)
                    x, y, z = list(zip(X, Y, Z))[ik]
                    ax.text(x, y, z, annotation)
                ax.view_init(30,30)
                plt.show()
            
                print("\n")
                print("## Select separation point ##   " + 'PCA separation' + "\n")
                for i in range(len(points_vessel)):
                    print("   ", i, " : point ", i)
                plt.show()


                target = int(input("-->  "))

                points_non_bas_P1 = points_vessel[:target]
                points_non_bas_P2 = points_vessel[target:]

            else:

                # Non Basilar side : Auto

                target = [points_Pcom[0], points_Pcom[points_Pcom.shape[0] - 1]]
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

                # Other side : Manual

                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="3d")
                ax.grid(False)

                ax.scatter(
                    points_bas[:, 0],
                    points_bas[:, 1],
                    points_bas[:, 2],
                    label="basilar",
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
                X, Y, Z = points_pca[:, 0], points_pca[:, 1], points_pca[:, 2]
                ax.plot(X, Y, Z, "--")
                ax.plot(X, Y, Z, "o")
                ax.legend()

                L = np.linspace(0, len(points_pca) - 1, 20)
                Lind = [int(np.floor(x)) for x in L]
                print(Lind)
                for ik in Lind:

                    annotation = "point {}".format(ik)
                    x, y, z = list(zip(X, Y, Z))[ik]
                    ax.text(x, y, z, annotation)

                plt.show()

                print("\n")
                print("## Select separation point ##  Basilar side PCA " + "\n")
                for i in range(len(points_pca)):
                    print("   ", i, " : point ", i)

                target = int(input("-->  "))

            points_bas_P1 = points_pca[:target]
            points_bas_P2 = points_pca[target:]

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.grid()
            ax.scatter(
                points_bas[:, 0],
                points_bas[:, 1],
                points_bas[:, 2],
                label="basilar",
            )

            # ax.scatter(points_bas_Pcom[:,0],points_bas_Pcom[:,1],points_bas_Pcom[:,2],label=' basilar side Pcom')
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
            plt.show()

            dpoints_divided = {}
            k = 0
            if points_bas.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = (
                    side_bas + "_BAS",
                    points_bas,
                )
                k += 1
            if points_bas_P1.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = (
                    side_bas + "_P1",
                    points_bas_P1,
                )
                k += 1
            if points_bas_P2.shape[0] != 0:
                dpoints_divided["points{}".format(k)] = (
                    side_bas + "_P2",
                    points_bas_P2,
                )
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
        dvectors["vectors{}".format(j)] = filename, geom.calculate_normal_vectors(points)
        j += 1
    return dpoints, dvectors


# %% Main


def _main_(pinfo, case, length):
    """
    This code initialize the dictionnary of the control points, taking into account the variation
    of the circle of Wilis of the patient.
    """

    dpoint_i = geom.create_dpoint(pinfo, case, length)

    # Step 2# Create the separated arteries

    dpoints_divI = division_ICA(pinfo, case, length)
    dpoints_divACA = division_ACA(pinfo, case, length)
    dpoints_divPCA = division_P_bas(pinfo, case, length)

    dpoints = dpoint_i.copy()

    # Step 3# Add the separated arteries

    dpoints = add_divided_arteries(dpoints, dpoints_divI)
    dpoints = add_divided_arteries(dpoints, dpoints_divACA)
    dpoints = add_divided_arteries(dpoints, dpoints_divPCA)

    # Step 4# Delete the old arteries

    dpoints, indices = delete_old_arteries(dpoints)

    dpoints, dvectors = createfinal_dicts(dpoints, indices)

    return dpoints, dvectors
