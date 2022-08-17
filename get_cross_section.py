# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:24:51 2022

@author: GALADRIEL_GUEST
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

#%%
os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

import geometry_slice as geom
import division_variation4 as variation

importlib.reload(geom)

#%%

# def get_slice(pinfo,case,i):
    
#     return slice_array

step = 10
num_cycle = 2
pinfo = 'pt2'
# dpoints_bas ,dvectors_bas = variation._main_(pinfo,'baseline',step)
# dpoints_vas ,dvectors_vas = variation._main_(pinfo,'vasospasm',step)
#  # Replace by load dict & return theese dict in main project
 


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
    v = V/np.linalg.norm(U)
    
    

    slice_proj = np.zeros((slice_rev.shape[0],2))
    for i in range(slice_rev.shape[0]):
        # x = slice_rev[i,:]
        # xprime = np.dot(U,x)
        # yprime = np.dot(V,x)
        
        xprime = np.dot(u,slice_rev[i,:]-origin)
        yprime = np.dot(v,slice_rev[i,:]-origin)
        
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
    Lslice=[]
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
            
            # one_rev = Slice_array.T[:,:1]
            # for k in one_rev.shape[0]:
            #     project_slice[k,:] = np.dot(one_rev[k,:],normal)  
                
            hull = ConvexHull(one_rev,qhull_options="Qc")
            A = hull.area
            # hull = ConvexHull(Slice_array.T)
            fig = plt.figure(figsize=(7, 7))
            plt.scatter(one_rev[:,0],one_rev[:,1],marker = 'o')
            for simplex in hull.simplices:
                plt.plot(one_rev[simplex, 0], one_rev[simplex, 1],'k-')
            plt.show()
            # plt.show()
            Lslice.append(A)
           
            # tslice.append((origin_slice,geom.find_radius(origin_slice,Slice_array.T)))
            print("   $$ Control point ", j, "Area :  = ", A)
            
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

    path = "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit/"
    os.chdir(path)
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

    return onlyfiles, indices



def compute_radius(pinfo,case,num_cycle,step, dpoints,dvectors,i):
    
    # importlib.reload(geom)
    
    
    os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

    import geometry_slice as geom

    importlib.reload(geom)


    dslice={}
    
    onlydat, indices_dat = get_list_files_dat(pinfo, case, num_cycle) # Get the dat files for the patient and its case


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
    
    
    # for i in range(len(dpoints)):
    #     dslice['slice{}'.format(i)] = compute_on_slice(data_file,i, dpoints, dvectors,pinfo,case)
        
    # return dslice
    return compute_on_slice_convex(data_file,i,dpoints,dvectors,pinfo,case)



def get_dCS(pinfo,case,num_cycle,step,dpoints,dvectors,ddist):

    
    Lvessel=['L_MCA','R_MCA','L_A1','L_A2','R_A1','R_A2','L_P1','L_P2','R_P1','R_P2','BAS','L_ICA','R_ICA']

    Lvessel_pth=[dpoints.get('points{}'.format(i))[0] for i in range(len(dpoints))]
    Lvessel_comp=Lvessel_pth.copy()
      
    
    Verity = np.zeros((len(Lvessel),len(Lvessel_comp)))
    
    for i in range(len(Lvessel)):
        for j in range(len(Lvessel_comp)):
            
            Verity[i,j] = (Lvessel[i] in Lvessel_comp[j])
    L_test = []
    L_ind = []
    for i in range(len(Lvessel)):
        for j in range(len(Lvessel_comp)):
            if Verity[i,j] == 1:
                L_test.append(i)
                L_ind.append(j)
                
    dCS = {}    
    for k in range(len(L_ind)):
        i = L_ind[k]
        dCS['slice{}'.format(i)] = compute_radius(pinfo,case,num_cycle,step, dpoints, dvectors, i)
    
    return dCS

    
    
    