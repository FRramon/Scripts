# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:24:51 2022

@author: GALADRIEL_GUEST

This script make a cleaning of the slices that are made in tecplot.
It extract the slices according to the set of points and vectors in input.
For each slice, the x,y,z coordinate of the points enclosed are extracted, and 
a projection of the slice in the slice plan if effectuated, to get a set of 2d points.
Then, the convex hull area is calculated from scipy, and the approximate area is computed from the 
alphashape module. From these two values, the convexity of the slice is computed.
The circularity of the slice is also extracted.
To avoid any irregular slice, all slices which present a product of circularity by convexity inferior to 0.9 are removed.
After the cleaning, the new set of points and vectors are returned.
"""

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
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


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
import alphashape
from descartes import PolygonPatch

import math
from itertools import combinations

#%%
os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

import geometry_slice as geom
import division_variation4 as variation
importlib.reload(geom)

#%%


def data_coor(data_file,pinfo,case):
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


def dist(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def find_closest(data_file,pinfo,case, origin, name):
    """


    Parameters
    ----------
    origin :coordinates of the control point on which one want to work

    Returns
    -------
    coordinates of the closest point to the control points, by minimazation with the euclidean distance

    """

    L = []
    coordinates_fluent = data_coor(data_file,pinfo,case)
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


def get_slice(data_file,origin, normal, name,pinfo,case):
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

    origin_s = find_closest(data_file,pinfo,case,origin, name)
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
    X_array = np.asarray(final_slice.values("X")[:])
    Y_array = np.asarray(final_slice.values("Y")[:])
    Z_array = np.asarray(final_slice.values("Z")[:])
    Slice_array= np.array((X_array,Y_array,Z_array))
    return Slice_array,origin_slice


def compute_on_slice(data_file,i_vessel, dpoints, dvectors,pinfo,case):
    """


    Parameters
    ----------
    i_vessel :index of the vessel.

    Returns
    -------
    Lpressure : list of the average pressure along the vessel.

    """

    name = dpoints.get("points{}".format(i_vessel))[0]

    Lavg, Lmin, Lmax = [], [], []

    points = dpoints.get("points{}".format(i_vessel))[1]
    vectors = dvectors.get("vectors{}".format(i_vessel))[1]
    dslice = {}
    n_ = points.shape[0]
    tslice=[]
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

        # fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_subplot(111, projection="3d")
        # ax.grid()

        for j in tqdm(range(len(points) - 1)):
            
            origin = points[j, :]
            normal = vectors[j, :]
            Slice_array,origin_slice = get_slice(
                data_file,origin, normal, name,pinfo,case
            )
            # print("   $$ Control point ", j, " : Pressure = ", avg_pressure)
            # Lmin.append(min_pressure)
            # Lavg.append(avg_pressure)
            # Lmax.append(max_pressure)
            # one_rev = Slice_array.T
            # hull = ConvexHull(one_rev)
            # hull.area
            # hull = ConvexHull(Slice_array.T)
            # fig = plt.figure(figsize=(7, 7))
            # ax = fig.add_subplot(111, projection="3d")
            # ax.scatter(Slice_array[1,:], Slice_array[1,:],Slice_array[2,:], 'o')
            # for simplex in hull.simplices:
            #     ax.plot(Slice_array[simplex, 0], Slice_array[simplex, 1], Slice_array[simplex,2],'k-')

            # plt.show()
           
            tslice.append((origin_slice,geom.find_radius(origin_slice,Slice_array.T)))
            print("   $$ Control point ", j, " : Radius = ", tslice[j][1])
            
    else:
        L = [0] * (len(points))
        tslice = (0,0)

    return tslice

def orthogonal_projection(Slice_array,origin,normal):
    slice_rev = Slice_array.T
    
    normal = normal/ np.linalg.norm(normal)
    
    U = np.array([-normal[1],normal[0],0])
    u = U/np.linalg.norm(U)
    
    V = np.array([-normal[0]*normal[2],-normal[1]*normal[2],normal[0]*normal[0]+normal[1]*normal[1]])
    v = V/np.linalg.norm(V)
    
    

    slice_proj = np.zeros((slice_rev.shape[0],2))
    for i in range(slice_rev.shape[0]):
        # x = slice_rev[i,:]
        # xprime = np.dot(U,x)
        # yprime = np.dot(V,x)
        
        xprime = np.dot(U,slice_rev[i,:]-origin)
        yprime = np.dot(V,slice_rev[i,:]-origin)
        
        slice_proj[i,0] = xprime
        slice_proj[i,1] = yprime
        
    return slice_proj

def another_orthogonal_projection(Slice_array,origin,normal):
    slice_rev = Slice_array.T
    
    # n = normal/ np.linalg.norm(normal)
    n = normal
    #n /= np.sqrt((n**2).sum())
    x = np.array([1,0,0])    
    x = x - np.dot(x, n) * n
    #x /= np.sqrt((x**2).sum())
    
    y = np.cross(n,x)
    

    slice_proj = np.zeros((slice_rev.shape[0],2))
    for i in range(slice_rev.shape[0]):
        # x = slice_rev[i,:]
        # xprime = np.dot(U,x)
        # yprime = np.dot(V,x)
        
        xprime = np.dot(x,slice_rev[i,:]-origin)
        yprime = np.dot(y,slice_rev[i,:]-origin)
        
        # print(xprime,yprime)
        
        slice_proj[i,0] = xprime
        slice_proj[i,1] = yprime
        
    return slice_proj


def compute_on_slice_convex(data_file,i_vessel, dpoints, dvectors,pinfo,case):
    """


    Parameters
    ----------
    i_vessel :index of the vessel.

    Returns
    -------
    Lpressure : list of the average pressure along the vessel.

    """

    name = dpoints.get("points{}".format(i_vessel))[0]

    Lavg, Lmin, Lmax = [], [], []

    points = dpoints.get("points{}".format(i_vessel))[1]
    vectors = dvectors.get("vectors{}".format(i_vessel))[1]
    dslice = {}
    n_ = points.shape[0]
    Lslice = np.zeros((n_-1,3))
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

        # fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_subplot(111, projection="3d")
        # ax.grid()

        for j in tqdm(range(len(points) - 1)):
            
            origin = points[j, :]
            normal = vectors[j, :]
            Slice_array,origin_slice = get_slice(
                data_file,origin, normal, name,pinfo,case
            )
            # print("   $$ Control point ", j, " : Pressure = ", avg_pressure)
            # Lmin.append(min_pressure)
            # Lavg.append(avg_pressure)
            # Lmax.append(max_pressure)
            one_rev = orthogonal_projection(Slice_array,origin,normal)
            
            
            # Convex Hull : Perimeter and Area
            hull = ConvexHull(one_rev)
            HullP = hull.area # Perimeter of the convex hull (N-1 dimension)
            HullArea = hull.volume # Area of the convex hull 
            
            
            # Alpha shape
            # distances = [dist(p1, p2) for p1, p2 in combinations(one_rev, 2)]
            # alpha_crit = sum(distances) / len(distances)
            # print(alpha_crit)
            Alpha = alphashape.alphashape(one_rev,500)
            Area = Alpha.area
            P = Alpha.length
            
            # Define morphological metrics :
            if P != 0:
                circularity = 4*np.pi*Area/(P*P)
            else:
                circularity = 0
            if HullArea !=0:
                convexity = Area/HullArea
            else: 
                convexity = 0
            
            
            Lslice[j,0] = Area
            Lslice[j,1] = circularity
            Lslice[j,2] = convexity
            
            if P!=0 and circularity !=0 :
                fig,ax = plt.subplots(figsize=(7, 7))
                ax.scatter(one_rev[:,0],one_rev[:,1],marker = 'o')
                
                for simplex in hull.simplices:
                    ax.plot(one_rev[simplex, 0], one_rev[simplex, 1],'k-')
                
                
                ax.add_patch(PolygonPatch(Alpha, alpha = 0.2))
                plt.show()
            print('\n')
            print("   $$ Control point ", j, "Circularity :  = ", circularity)
            # print("   $$ Control point ", j, "Convex Hull Circularity :  = ", 4*np.pi*HullArea/(HullP*HullP))
            print("   $$ Control point ", j, "Convexity :  = ",convexity)
            print("   $$ Control point ", j, "Area :  = ", Area)
            
            

            
    else:
        Lslice = [0] * (len(points))
        

    return Lslice

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

    pathwd = "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit"
    # mesh_size = "5"
    # list_files_dir = os.listdir("N:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit")
    # for files_dir in list_files_dir:
    #     if "mesh_size_" + mesh_size in files_dir:
    #         pathwd = pathwd + "/" + files_dir
    os.chdir(pathwd)
    onlyfiles = []
    for file in glob.glob("*.dat"):
        if pinfo + "_" + case + "_cycle" + num_cycle in file:
            onlyfiles.append(file)
    indices = [l[13:-4] for l in onlyfiles]

    for i in range(len(indices)):
        newpath = (
            "N:/vasospasm/pressure_pytec_scripts/plots_8_4/"
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

    return onlyfiles, indices,pathwd



def compute_radius(pinfo,case,num_cycle, dpoints,dvectors,i):
    
    
    
    os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

    import geometry_slice as geom

    importlib.reload(geom)


    dslice={}
    
    onlydat, indices_dat,pathwd = get_list_files_dat(pinfo, case, num_cycle) # Get the dat files for the patient and its case
    print(pathwd)

    filename = onlydat[0]

    # print(
    #     " ############  Step 2 : Connection to Tecplot  ############ Time step : ",
    #     i,
    #     "\n",
    # )
    logging.basicConfig(level=logging.DEBUG)

    # To enable connections in Tecplot 360, click on:
    #   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

    tp.session.connect()
    tp.new_layout()
    frame = tp.active_frame()

    # dir_file = (
    #     "N:/vasospasm/"
    #     + pinfo
    #     + "/"
    #     + case
    #     + "/3-computational/hyak_submit/"
    #     + filename
    # )
    dir_file = pathwd + '/' + filename
   
    # data_file = tp.data.load_fluent(
    #     case_filenames=[
    #         "N:/vasospasm/"
    #         + pinfo
    #         + "/"
    #         + case
    #         + "/3-computational/hyak_submit/"
    #         + pinfo
    #         + "_"
    #         + case
    #         + ".cas"
    #     ],
    #     data_filenames=[dir_file],
    # )
    
    # data_file = tp.data.load_fluent(
    #     case_filenames=[
    #         pathwd + '/' 
    #         + pinfo
    #         + "_"
    #         + case
    #         + ".cas"
    #     ],
    #     data_filenames=[dir_file],
    # )
    data_file = tp.data.load_fluent(
        case_filenames=[
            pathwd + '/' 
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
    
    
    # for i in range(len(dpoints)):
    #     dslice['slice{}'.format(i)] = compute_on_slice(data_file,i, dpoints, dvectors,pinfo,case)
        
    # return dslice
    return compute_on_slice_convex(data_file,i,dpoints,dvectors,pinfo,case)



def get_dCS(pinfo,case,num_cycle,dpoints,dvectors):

    
    Lvessel=['L_MCA','R_MCA','L_A1','L_A2','R_A1','R_A2','L_P1','L_P2','R_P1','R_P2','BAS','L_ICA','R_ICA']

    Lvessel_pth=[dpoints.get('points{}'.format(i))[0] for i in range(len(dpoints))]
    Lvessel_comp=Lvessel_pth.copy()
      
    
    Verity = np.zeros((len(Lvessel),len(Lvessel_comp)))
    
    for i in range(len(Lvessel)):
        for j in range(len(Lvessel_comp)):
            
            Verity[i,j] = (Lvessel[i] in Lvessel_comp[j])
    L_test = []
    L_ind = []
    print(Verity)
    for i in range(len(Lvessel)):
        for j in range(len(Lvessel_comp)):
            if Verity[i,j] == 1:
                L_test.append(i)
                L_ind.append(j)
                
    dCS = {}    
    for k in range(len(L_ind)):
        i = L_ind[k]
        dCS['slice{}'.format(i)] = compute_radius(pinfo,case,num_cycle, dpoints, dvectors, i)
    
    return dCS,L_ind

    
    
def morphometric_cleaning(dCS,L_ind,dpoints,dvectors):
    # dCS structure  : 
        #dict of the slices of the vessel concerned by the pressure compute 
        # Inside a point of a vessel : 3,1 list : Area, convexity, circularity:
        # identify the slices which have a weak convexity*circularity criterion
        # Create new dpoints and dvectors with only the points and normal which are reliable
        # Use these two to compute pressure.
        
        
    n_dCS,n_dpoints,n_dvectors = {},{},{}
    #(Add everything to the main at some point)
    for i in range(len(dCS)):
        L_toremove = []
        array_vessel = dCS.get("slice{}".format(L_ind[i]))
        name_vessel = dpoints.get("points{}".format(L_ind[i]))[0]
        for j in range(array_vessel.shape[0]):
            criterion = array_vessel[j][1] * array_vessel[j][2]
            if criterion < 0.9:
                print(i,j)
                print(array_vessel[j][1])
                print(array_vessel[j][2])

                
                L_toremove.append(j)
        len_vessel = dpoints.get("points{}".format(L_ind[i]))[1].shape[0]

        L_tokeep = [i for i in range(len_vessel) if i not in L_toremove]
        length_new = len_vessel-len(L_toremove)
        new_points = np.zeros((length_new,3))
        new_vectors =  np.zeros((length_new-1,3))
        new_CS =  np.zeros((length_new-1,3))
        index = 0
        for k in L_tokeep:
            new_points[index,:] = dpoints.get("points{}".format(L_ind[i]))[1][k,:]
            
            index+=1
        index=0
        for k in L_tokeep[:-1]:
            new_vectors[index,:] = dvectors.get("vectors{}".format(L_ind[i]))[1][k,:]
            new_CS[index,:] = dCS.get("slice{}".format(L_ind[i]))[k,:]

            index+=1
            
        n_dCS['slice{}'.format(i)] = name_vessel,new_CS
        n_dpoints['points{}'.format(i)] = name_vessel,new_points
        n_dvectors['vectors{}'.format(i)] = name_vessel,new_vectors
        
    return n_dCS,n_dpoints,n_dvectors


# points_test = n_dp.get("points0")[1]

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# ax.grid()

# ax.scatter(points_test[:,0],points_test[:,1],points_test[:,2])
            
        
# pinfo = 'pt7'
# case = 'baseline'
# num_cycle = 2
# dpoints_d = dpoints_7_bas
# dvectors_d = dvectors_7_bas
# dCS, L_ind = cross_section.get_dCS(pinfo, case, num_cycle, dpoints_d, dvectors_d)
# n_dcs_7_bas,dpoints_u,dvectors_u = cross_section.morphometric_cleaning(dCS, L_ind, dpoints_d, dvectors_d)

# case = 'vasospasm'
# dpoints_d = dpoints_7_vas
# dvectors_d = dvectors_7_vas
# dCS, L_ind = cross_section.get_dCS(pinfo, case, num_cycle, dpoints_d, dvectors_d)
# n_dcs_7_vas,dpoints_u,dvectors_u = cross_section.morphometric_cleaning(dCS, L_ind, dpoints_d, dvectors_d)

# for i in range(len(n_dcs_7_bas)):
#     fig, ax = plt.subplots()
#     data_bas = n_dcs_7_bas.get('slice{}'.format(i))[1][:,0]
#     data_vas = n_dcs_7_vas.get('slice{}'.format(i))[1][:,0]
#     print(data_bas)
#     abscisse = np.linspace(0,len(data_bas),len(data_bas))
#     abscisse2 = np.linspace(0,len(data_vas),len(data_vas))
#     ax.plot(abscisse,data_bas,label = "baseline case")
#     ax.plot(abscisse2,data_vas, label= "vasospasm case")
#     ax.set_title(n_dcs_7_bas.get("slice{}".format(i)[0]))
#     ax.legend()
#     plt.show()
            
# plt.show()        
            
            
            

    
    