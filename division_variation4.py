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
        root = ET.fromstring(
            re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

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



def find_number_of_steps(points_vessel,radius):
    
    
    # Convert the radius into the equivalent of steps in the vessel coordinates array
    
    vect_vessel=calculate_normal_vectors(points_vessel)
    norms_vessel=calculate_norms(vect_vessel)
    
    # Compute the norms from 0 to i, i variating between 0 and len(vessel)
    L_dist_along=[np.sum(norms_vessel[0:i]) for i in range(norms_vessel.shape[0]) ]
    # Compare the previous norms and the radius
    L_compare_r=[abs(L_dist_along[i]-radius) for i in range(len(L_dist_along))]
    # Select the index of the minimum distance, which correspond to the indice to remove.
    step_vessel=L_compare_r.index(min(L_compare_r))
  
    # return step_bas,step_lsc,step_rsc
    
    return step_vessel

# def get_center_radius(fname,pinfo,case):
             
#         os.chdir('N:/vasospasm/'+pinfo +'/'+ case +'/1-geometry/'+ pinfo + '_' + case + '_segmentation_no_vti/Segmentations')
        
        
#         fname='L_ICA_MCA.ctgr'
#         with open(fname) as f:
#             xml = f.read()
#             root = ET.fromstring(
#                 re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")
           
#         # find the branch of the tree which contains control points
        
#         branch=root[1][0]
#         n_points = len(branch)
#         dsurfaces={}
#         for j in range(1,n_points):
#             s_points=np.zeros((len(branch[j][2]),3))
#             for i in range(0,len(branch[j][2])):
#                 leaf = branch[j][2][i].attrib
            
#                 s_points[i][0] = float(leaf.get('x')) * 0.001
#                 s_points[i][1] = float(leaf.get('y')) * 0.001
#                 s_points[i][2] = float(leaf.get('z')) * 0.001
         
#             dsurfaces['surface{}'.format(j)]=s_points
        
#         dcenter={}
#         for j in range(1,len(dsurfaces)+1):
           
#             L=np.asarray(dsurfaces.get('surface{}'.format(j)))
        
#             center_x = np.mean(L[:,0])
#             center_y = np.mean(L[:,1])
#             center_z = np.mean(L[:,2])
        
#             center=np.array((center_x,center_y,center_z))
        
#             Lradius=[]
#             for i in range(L.shape[0]):
#                 Lradius.append(np.linalg.norm(center-L[i,:]))
#             radius=max(Lradius)
        
#             dcenter['center{}'.format(j)]=center,radius
        
#         return dcenter


def get_center_radius_ulti(fname,pinfo,case):
             
        os.chdir('N:/vasospasm/'+pinfo +'/'+ case +'/1-geometry/'+ pinfo + '_' + case + '_segmentation_no_vti/Segmentations')
        
        
        fname='L_ICA_MCA.ctgr'
        with open(fname) as f:
            xml = f.read()
            root = ET.fromstring(
                re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")
           
        # find the branch of the tree which contains control points
        
        branch=root[1][0]
        n_points = len(branch)
        dsurfaces={}
        for j in range(1,n_points):
            s_points=np.zeros((len(branch[j][2]),3))
            for i in range(0,len(branch[j][2])):
                leaf = branch[j][2][i].attrib
            
                s_points[i][0] = float(leaf.get('x')) * 0.001
                s_points[i][1] = float(leaf.get('y')) * 0.001
                s_points[i][2] = float(leaf.get('z')) * 0.001
         
            dsurfaces['surface{}'.format(j)]=s_points
        
        dcenter={}
        
           
        Lstart=np.asarray(dsurfaces.get('surface{}'.format(1)))
        Lend=np.asarray(dsurfaces.get('surface{}'.format(len(dsurfaces))))

        center_x1 = np.mean(Lstart[:,0])
        center_y1 = np.mean(Lstart[:,1])
        center_z1 = np.mean(Lstart[:,2])
        
        center_x2 = np.mean(Lend[:,0])
        center_y2 = np.mean(Lend[:,1])
        center_z2 = np.mean(Lend[:,2])
        
        center1=np.array((center_x1,center_y1,center_z1))
        center2=np.array((center_x2,center_y2,center_z2))

        
        Lradius=[]
        for i in range(Lstart.shape[0]):
            Lradius.append(np.linalg.norm(center1-Lstart[i,:]))
        radius1=max(Lradius)
        
        Lradius=[]
        for i in range(Lend.shape[0]):
            Lradius.append(np.linalg.norm(center2-Lend[i,:]))
        radius2=max(Lradius)
        
        
        dcenter['center{}'.format(1)]=center1,radius1
        dcenter['center{}'.format(2)]=center2,radius2

        
        return dcenter


    
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

    if pinfo == 'pt2':
        folder = '_segmentation_no_vti'
    else:
        folder = '_segmentation'
    pathpath = 'N:/vasospasm/' + pinfo + '/' + case+'/1-geometry/' + \
        pinfo + '_' + case + folder + '/paths'

    os.chdir(pathpath)
    onlyfiles = []
    for file in glob.glob("*.pth"):
        onlyfiles.append(file)
    i = 0
    dpoint_i = {}
    for file in onlyfiles:

        filename = file[:-4]
        dpoint_i["points{}".format(
            i)] = filename, get_spline_points(file, step)
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

    #LOAD .pth files (Control points)

    if pinfo == 'pt2':
        folder = '_segmentation_no_vti'
    else:
        folder = '_segmentation'
    pathpath = 'N:/vasospasm/' + pinfo + '/' + case+'/1-geometry/' + \
        pinfo + '_' + case + folder + '/paths'

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
            
    # LOAD .ctgr files (center, radius) 
            
    pathctgr='N:/vasospasm/'+ pinfo + '/' + case + '/1-geometry/' +  pinfo + '_' + case + '_segmentation_no_vti/Segmentations'
    os.chdir(pathctgr)
    
    filesctgr = []
    for file in glob.glob("*.ctgr"):
        filesctgr.append(file)
    for files in filesctgr:
        if "L_ACA" in files:
            center_LACA = get_center_radius_ulti(files, pinfo,case)
        if "R_ACA" in files:
            center_RACA =get_center_radius_ulti(files, pinfo,case)
        # if "L_ICA_MCA" in files:
        #     center_LICAMCA = get_center_radius(files, pinfo,case)
        # if "R_ICA_MCA" in files:
        #     center_RICAMCA = get_center_radius(files, pinfo,case)
    
            

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
        limin_final=limin
    
    points_LICA = points_LICAMCA[:limin_final]
    points_LMCA = points_LICAMCA[limin_final:]

    #Definition of the indices to truncate (the equivalent of a radius of laca on each side)

    if limin <= len(lnorms_end):
        indice_LICA=find_number_of_steps(points_LICA, center_LACA.get('center1')[1])
        indice_LMCA=find_number_of_steps(points_LMCA, center_LACA.get('center1')[1])
    else:
        indice_LICA=find_number_of_steps(points_LICA, center_LACA.get('center2')[1])
        indice_LMCA=find_number_of_steps(points_LMCA, center_LACA.get('center2')[1])

    points_LICA=points_LICA[:points_LICA.shape[0]-indice_LICA]
    points_LMCA=points_LMCA[indice_LMCA:]

    


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
        indice_RICA=find_number_of_steps(points_RICA, center_RACA.get('center1')[1])
        indice_RMCA=find_number_of_steps(points_RMCA, center_RACA.get('center1')[1])
    else:
        indice_RICA=find_number_of_steps(points_RICA, center_RACA.get('center2')[1])
        indice_RMCA=find_number_of_steps(points_RMCA, center_RACA.get('center2')[1])

    points_RICA=points_RICA[:points_RICA.shape[0]-indice_RICA]
    points_RMCA=points_RMCA[indice_RMCA:]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid()

    ax.scatter(points_LICA[:, 0], points_LICA[:, 1],
               points_LICA[:, 2], c='b', label='LEFT ICA')
    ax.scatter(points_RICA[:, 0], points_RICA[:, 1],
                points_RICA[:, 2], c='k', label='RIGHT ICA ')
    ax.scatter(points_RMCA[:, 0], points_RMCA[:, 1],
                points_RMCA[:, 2], c='r', label='RIGHT MCA')
    ax.scatter(points_LMCA[:, 0], points_LMCA[:, 1],
               points_LMCA[:, 2], c='g', label='LEFT MCA')
    ax.scatter(points_LACA[:, 0], points_LACA[:, 1],
                points_LACA[:, 2], label='LEFT ACA ')
    ax.scatter(points_RACA[:, 0], points_RACA[:, 1],
                points_RACA[:, 2], label='RIGHT ACA ')

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


def division_RP(pinfo, case, step):
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


    if pinfo == 'pt2':
        folder = '_segmentation_no_vti'
    else:
        folder = '_segmentation'
    pathpath = 'N:/vasospasm/' + pinfo + '/' + case+'/1-geometry/' + \
        pinfo + '_' + case + folder + '/paths'

    os.chdir(pathpath)
    onlyfiles = []

    vessel='R_Pcom_PCA'

    if vessel[0] == 'L':
        other_side = 'R'
    else:
        other_side = 'L'

    for file in glob.glob("*.pth"):
        onlyfiles.append(file)
    for files in onlyfiles:
        if 'Pcom_PCA' in files:
            points_vessel = get_spline_points(files, step)
        if 'BAS_PCA' in files:
            points_bas = get_spline_points(files, step)
        if other_side + '_Pcom' in files:
            points_pcom = get_spline_points(files, step)
            
    # LOAD .ctgr files (center, radius) 
             
    pathctgr='N:/vasospasm/'+ pinfo + '/' + case + '/1-geometry/' +  pinfo + '_' + case + '_segmentation_no_vti/Segmentations'
    os.chdir(pathctgr)
     
    filesctgr = []
    for file in glob.glob("*.ctgr"):
        filesctgr.append(file)
    for files in filesctgr:
        if other_side + "_Pcom" in files:
            center_pcom = get_center_radius_ulti(files, pinfo,case)
       
             

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.scatter(points_bas[:, 0], points_bas[:, 1],
               points_bas[:, 2], c='k', label=other_side + '_BAS_PCA')
    ax.scatter(points_pcom[:, 0], points_pcom[:, 1],
               points_pcom[:, 2], c='b', label=other_side + '_Pcom')
    X, Y, Z = points_vessel[:, 0], points_vessel[:, 1], points_vessel[:, 2]
    ax.plot(X, Y, Z, '--')
    ax.plot(X, Y, Z, 'o', label=vessel)

    ax.legend()
    plt.title(vessel)
    # i_point=0
    step_point = int((len(points_vessel)) / (len(points_vessel) / 2))

    L = np.linspace(0, len(points_vessel) - 1, 20)
    Lind = [int(np.floor(x)) for x in L]
    print(Lind)
    for ik in Lind:

        annotation = 'point {}'.format(ik)
        x, y, z = list(zip(X, Y, Z))[ik]
        ax.text(x, y, z, annotation)

   
    plt.show()

    print('\n')
    print('## Select separation point ##   ' + vessel[:-4] + '\n')
    for i in range(len(points_vessel)):
        print('   ', i, ' : point ', i)

    target = int(input('-->  '))
    

    points_1 = points_vessel[target:]
    points_2 = points_vessel[:target]
    
    indice_p2=find_number_of_steps(points_2, center_pcom.get('center2')[1])
    points_1=points_1[indice_p2:]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.scatter(points_bas[:, 0], points_bas[:, 1],
               points_bas[:, 2], label=other_side + '_BAS_PCA')
    ax.scatter(points_pcom[:, 0], points_pcom[:, 1],
               points_pcom[:, 2], label=other_side + '_Pcom')
    ax.scatter(points_1[:, 0], points_1[:, 1],
               points_1[:, 2], label=vessel[0] + 'PCA P2')
    ax.scatter(points_2[:, 0], points_2[:, 1],
                points_2[:, 2], label=vessel[0] + '_Pcom')

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


    if pinfo == 'pt2':
        folder = '_segmentation_no_vti'
    else:
        folder = '_segmentation'
    pathpath = 'N:/vasospasm/' + pinfo + '/' + case+'/1-geometry/' + \
        pinfo + '_' + case + folder + '/paths'

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

    pathctgr='N:/vasospasm/'+ pinfo + '/' + case + '/1-geometry/' +  pinfo + '_' + case + '_segmentation_no_vti/Segmentations'
    os.chdir(pathctgr)
       
    filesctgr = []
    for file in glob.glob("*.ctgr"):
        filesctgr.append(file)
    for files in filesctgr:
        if "Acom" in files:
            center_Acom = get_center_radius_ulti(files, pinfo,case)
            

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
        limin_final = limin - len(Lnorms_end)
    else: 
        limin_final = limin
        
    points_LA1 = points_LACA[:limin_final]
    points_LA2 = points_LACA[limin_final:]
    
    # REMOVE THE RAIDUS OF THE INTESECTING VESSEL
    
    if limin <= len(Lnorms_end):
        indice_LA1=find_number_of_steps(points_LA1, center_Acom.get('center1')[1])
        indice_LA2=find_number_of_steps(points_LA2, center_Acom.get('center1')[1])
    else:
        indice_LA1=find_number_of_steps(points_LA1, center_Acom.get('center2')[1])
        indice_LA2=find_number_of_steps(points_LA2, center_Acom.get('center2')[1])
     
        
    points_LA1=points_LA1[:points_LA1.shape[0]-indice_LA1]
    points_LA2=points_LA2[indice_LA1:]


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
        rimin_final = rimin - len(Rnorms_end)
    else: 
        rimin_final = rimin
        
    points_RA1 = points_RACA[:rimin_final]
    points_RA2 = points_RACA[rimin_final:]
    
    # REMOVE THE RADIUS OF THE INTERSECTING VESSEL 
    
    if rimin <= len(Rnorms_end):
        indice_RA1=find_number_of_steps(points_RA1, center_Acom.get('center1')[1])
        indice_RA2=find_number_of_steps(points_RA2, center_Acom.get('center1')[1])
    else:
        indice_RA1=find_number_of_steps(points_RA1, center_Acom.get('center2')[1])
        indice_RA2=find_number_of_steps(points_RA2, center_Acom.get('center2')[1])
     
        
    points_RA1=points_RA1[:points_RA1.shape[0]-indice_RA1]
    points_RA2=points_RA2[indice_RA1:]
    
    

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid()

    ax.scatter(points_Acom[:, 0], points_Acom[:, 1],
               points_Acom[:, 2], label='Acom')
    ax.scatter(points_RA1[:, 0], points_RA1[:, 1],
               points_RA1[:, 2], label='RIGHT ACA A1')
    ax.scatter(points_RA2[:, 0], points_RA2[:, 1],
               points_RA2[:, 2], label='RIGHT ACA A2')
    ax.scatter(points_LA1[:, 0], points_LA1[:, 1],
               points_LA1[:, 2], label='LEFT ACA A1')
    ax.scatter(points_LA2[:, 0], points_LA2[:, 1],
               points_LA2[:, 2], label='LEFT ACA A2')

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

    pathctgr='N:/vasospasm/'+ pinfo + '/' + case + '/1-geometry/' +  pinfo + '_' + case + '_segmentation_no_vti/Segmentations'
    os.chdir(pathctgr)
         
    filesctgr = []
    for file in glob.glob("*.ctgr"):
        filesctgr.append(file)
    for files in filesctgr:
        if "BAS_PCA" in files:
            center_BAS = get_center_radius_ulti(files, pinfo,case)
            side_bas = files[0]
            for subfiles in filesctgr:
                if side_bas + "_Pcom" in subfiles:
                    center_pcom=get_center_radius_ulti(subfiles,pinfo,case)


    if pinfo == 'pt2':
        folder = '_segmentation_no_vti'
    else:
        folder = '_segmentation'
    pathpath = 'N:/vasospasm/' + pinfo + '/' + case+'/1-geometry/' + \
        pinfo + '_' + case + folder + '/paths'

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

                if side_bas + '_Pcom' in subfile:
                    points_target = get_spline_points(subfile, step)

                if 'Pcom_PCA' in subfile:
                    points_otherside = get_spline_points(subfile, step)

            target = [points_target[0],
                      points_target[points_target.shape[0] - 1]]
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


            # DIVISION BAS/P1 AND P2

            points_P2 = points_bas_pca[limin_final:]
            points_basP1 = points_bas_pca[:limin_final]
            
            # REMOVE A RADIUS OF THE INTERSECTING VESSEL 
            
            if limin <= len(lnorms_end):
                indice_basP1 = find_number_of_steps(points_basP1, center_pcom.get('center1')[1])
                indice_P2 = find_number_of_steps(points_P2, center_pcom.get('center1')[1])

            else:
                indice_basP1 = find_number_of_steps(points_basP1, center_pcom.get('center2')[1])
                indice_P2 = find_number_of_steps(points_P2, center_pcom.get('center2')[1])
             
                
            print(indice_P2)
            print(indice_basP1)
            points_P2 = points_P2[indice_P2:]
            points_basP1 = points_basP1[:points_basP1.shape[0]-indice_basP1]
        
        

            # DIVISION BAS & P1

            # PLOT AND DIVIDE MANUALLY

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.grid(False)
            ax.scatter(points_otherside[:, 0], points_otherside[:,
                       1], points_otherside[:, 2], label='PCA_Pcom')

            ax.scatter(points_P2[:, 0], points_P2[:, 1],
                       points_P2[:, 2], c='b', label='P2')
            ax.scatter(points_target[:, 0], points_target[:, 1],
                       points_target[:, 2], c='g', label=side_bas + '_Pcom')
            X, Y, Z = points_basP1[:,
                                   0], points_basP1[:, 1], points_basP1[:, 2]
            ax.scatter(X, Y, Z, 'o', label='BAS + P1')

            ax.legend()
            plt.title('BAS + P1 Division')
            L = np.linspace(0, points_basP1.shape[0] - 1, 20)
            Lind = [int(np.floor(x)) for x in L]
            print(Lind)
            for ik in Lind:

                annotation = 'point {}'.format(ik)
                x, y, z = list(zip(X, Y, Z))[ik]
                ax.text(x, y, z, annotation)
            plt.show()

            print('\n')
            print('## Select separation point ##   ' + 'BAS_P1' + '\n')
            for i in range(len(points_basP1)):
                print('   ', i, ' : point ', i)

            target = int(input('-->  '))

            plt.show()
            points_bas = points_basP1[:target]
            points_P1 = points_basP1[target:]
            
            # REMOVE RADIUS OF BAS IN THE LEFT P1
            
            indice_bas = find_number_of_steps(points_P1, center_BAS.get('center1')[1])
            
            points_P1=points_P1[indice_bas:]

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.grid()

            ax.scatter(points_target[:, 0], points_target[:, 1],
                       points_target[:, 2], c='k', label=' basilar side Pcom')
            ax.scatter(points_bas[:, 0], points_bas[:, 1],
                       points_bas[:, 2], c='g', label='Basilar')
            ax.scatter(points_P1[:, 0], points_P1[:, 1],
                       points_P1[:, 2], c='b', label='PCA P1')
            ax.scatter(points_P2[:, 0], points_P2[:, 1],
                       points_P2[:, 2], c='r', label='PCA P2')

            ax.legend()
            plt.show()

            dpoints_divided = {}
            k = 0
            if points_bas.shape[0] != 0:
                dpoints_divided["points{}".format(
                    k)] = side_bas + "_BAS", points_bas
                k += 1
            if points_P1.shape[0] != 0:
                dpoints_divided["points{}".format(
                    k)] = side_bas + "_P1", points_P1
                k += 1
            if points_P2.shape[0] != 0:
                dpoints_divided["points{}".format(
                    k)] = side_bas + "_P2", points_P2
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
    indices = [i for i in range(
        len(dpoint_i) + len(I_supp)) if i not in I_supp]
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
        newpath = 'N:/vasospasm/pressure_pytec_scripts/plots_c/' + \
            pinfo + '/' + case+'/cycle_' + num_cycle + '/plot_' + indices[i]
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    return onlyfiles, indices


def save_dict(dico, name):
    '''


    Parameters
    ----------
    dico : dictionary one wants to save
    name : str. path + name of the dictionary

    Returns
    -------
    None.

    '''

    with open(name + '.pkl', 'wb') as f:
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
        dvectors["vectors{}".format(
            j)] = filename, calculate_normal_vectors(points)
        j += 1
    return dpoints, dvectors


# %% Main

def _main_(pinfo, case, step):

    
    print('patient info : ' ,pinfo + case )
    dpoint_i = create_dpoint(pinfo, case, step)

    # Step 2# CREATE NEW DIVIDED VESSELS

    dpoints_divI = division_ICA(pinfo, case, step)
    dpoints_divACA = division_A(pinfo, case, step)
    dpoints_divLPCA = new_division_P_bas(pinfo, case, step)
    dpoints_divRP = division_RP(pinfo, case, step)

    dpoints = dpoint_i.copy()

    # Step 3# DELETE THE OLD VESSELS

    dpoints = add_divided_arteries(dpoints, dpoints_divI)

    dpoints = add_divided_arteries(dpoints, dpoints_divACA)

    dpoints = add_divided_arteries(dpoints, dpoints_divLPCA)
    dpoints = add_divided_arteries(dpoints, dpoints_divRP)

    dpoints, indices = delete_old_arteries(dpoints)

    dpoints, dvectors = createfinal_dicts(dpoints, indices)

    return dpoints, dvectors
