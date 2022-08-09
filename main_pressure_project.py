# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:17:42 2022

@author: Francois


"""


# %% Imports


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


# %% Scripts as modules

os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

import geometry_slice as geom

# %% Functions


def get_distance_along(i_vessel, i_dat, dpressure, dvectors, dpoints,pinfo,case):
    # Change name function?
    """


    Parameters
    ----------
    dvectors : dictionary of the normal vectors in the patient's circle of Willis

    Returns
    -------
    ddist : dictionnary of the distance along the vessels (use to plot the pressure later)

    """
    dnorms = {}
    ddist = {}
    dvectors = {}
    onlydat, indices_dat = get_list_files_dat(pinfo, case, num_cycle)
    print(indices_dat[i_dat])
    # Inverse the order of the points if the pressure is not decreasing

    if (
        dpressure.get("{}".format(indices_dat[i_dat])).get(
            "pressure{}".format(i_vessel)
        )[1][0]
        == 1
    ):  # Check if it is to reverse (marked 1, 0 if not)
        array_points = dpoints.get("points{}".format(i_vessel))[1][
            ::-1
        ]  # reverse the array
        dvectors["vectors{}".format(i_vessel)] = dpoints.get(
            "points{}".format(i_vessel)
        )[0], geom.calculate_normal_vectors(
            array_points
        )  # Compute new vectors
    else:
        array_points = dpoints.get("points{}".format(i_vessel))[1]
        dvectors["vectors{}".format(i_vessel)] = dpoints.get(
            "points{}".format(i_vessel)
        )[0], geom.calculate_normal_vectors(array_points)

    dnorms["norms{}".format(i_vessel)] = dvectors.get("vectors{}".format(i_vessel))[
        0
    ], geom.calculate_norms(dvectors.get("vectors{}".format(i_vessel))[1])

    n_norms = dnorms.get("norms{}".format(i_vessel))[1].shape[0]

    dist = np.zeros((n_norms + 1, 1))
    for j in range(1, n_norms + 1):
        dist[j] = dnorms.get("norms{}".format(i_vessel))[1][:j, :].sum()
        ddist["dist{}".format(i_vessel)] = (
            dnorms.get("norms{}".format(i_vessel))[0],
            dist,
        )
    return ddist


def get_distance(vectors):
    norms = geom.calculate_norms(vectors)
    dist = np.zeros((norms.shape[0] + 1, 1))
    for j in range(1, norms.shape[0] + 1):
        dist[j] = norms[:j, :].sum()

    return dist


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


def load_dict(name):
    """


    Parameters
    ----------
    name : str. path + name of the dictionary one wants to load

    Returns
    -------
    b : the loaded dictionary

    """
    with open(name + ".pkl", "rb") as handle:
        b = pickle.load(handle)
    return b


def get_variation(pinfo, case):

    dinfo = scipy.io.loadmat(
        "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/case_info.mat"
    )
    variation = int(dinfo.get("variation_input"))
    if (variation==3) or (variation==4):
        variation = 4
    elif (variation==5) or (variation==6):
        variation = 5
    elif (variation==7) or (variation==8):
        variation = 7

    return variation




# %% Functions Tecplot

def plot_cross_section(pinfo,name):
    
    dradius_vas={}
    dradius_bas={}
    
    folder = "_segmentation"
    pathpath = (
        "N:/vasospasm/"
        + pinfo
        + "/"
        + "baseline"
        + "/1-geometry/"
        + pinfo
        + "_"
        + "baseline"
        + folder
        + "/Segmentations"
    )

    os.chdir(pathpath)
    onlyfiles = []
    for file in glob.glob("*.ctgr"):
        onlyfiles.append(file)
    for file in onlyfiles:
        if name in file:
            dradius_bas['{}'.format(name)]=geom.get_center_radius(file, pinfo, 'baseline')
        
    folder = "_segmentation"
    pathpath = (
        "N:/vasospasm/"
        + pinfo
        + "/"
        + "vasospasm"
        + "/1-geometry/"
        + pinfo
        + "_"
        + "vasospasm"
        + folder
        + "/Segmentations"
    )

    os.chdir(pathpath)
    onlyfiles = []
    for file in glob.glob("*.ctgr"):
        onlyfiles.append(file)
    for file in onlyfiles:
        if name in file:
            dradius_vas['{}'.format(name)]=geom.get_center_radius(file, pinfo, 'vasospasm')
        
        
    
    dict_interb=dradius_bas.get(name)
    Array_control_points_bas=np.zeros((len(dict_interb),3))
    Array_radius_bas=np.zeros((len(dict_interb),1))

    for i in range(1,len(dict_interb)+1):
        points,radius=dict_interb.get('center{}'.format(i))
        Array_control_points_bas[i-1,:]=points
        Array_radius_bas[i-1]=radius
    print('bas : ',Array_radius_bas.shape)
    
    dict_interv=dradius_vas.get(name)
    Array_control_points_vas=np.zeros((len(dict_interv),3))
    Array_radius_vas=np.zeros((len(dict_interv),1))

    for i in range(1,len(dict_interv)+1):
        points,radius=dict_interv.get('center{}'.format(i))
        Array_control_points_vas[i-1,:]=points
        Array_radius_vas[i-1]=radius
    print('vas : ',Array_radius_vas.shape)
        
    vect_bas=geom.calculate_normal_vectors(Array_control_points_bas)
    vect_vas=geom.calculate_normal_vectors(Array_control_points_vas)
    
    Norms_bas= geom.calculate_norms(vect_bas)
    Norms_vas =  geom.calculate_norms(vect_vas)
    Norms_bas_f=np.zeros((Norms_bas.shape[0]+1,1))
    Norms_vas_f=np.zeros((Norms_vas.shape[0]+1,1))
    
   
    Norms_bas_list=[float(sum(Norms_bas[:i])) for i in range(Norms_bas.shape[0])]
    Norms_vas_list=[float(sum(Norms_vas[:i])) for i in range(Norms_vas.shape[0])]

    
    # Xb=np.linspace(0,len(dict_interb),len(dict_interb))
    # Xv=np.linspace(0,len(dict_interv),len(dict_interv))

    plt.plot(Norms_bas_list,Array_radius_bas[:-1], label= 'radius | baseline ')
    plt.plot(Norms_vas_list,Array_radius_vas[:-1], label = 'radius | vasospasm')
    plt.ylabel('radius (m)')
    plt.xlabel('Distance along the vessel')
    plt.title("Cross section along the " + name)
    plt.legend()
    
    plt.show()
    



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


def get_pressure(data_file,origin, normal, name):
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
    min_pressure = np.min(final_slice.values("Pressure")[:])
    avg_pressure = np.mean(final_slice.values("Pressure")[:])
    max_pressure = np.max(final_slice.values("Pressure")[:])

    return min_pressure, avg_pressure, max_pressure


def compute_along(data_file,i_vessel, dpoints, dvectors):
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

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.grid()

        for j in tqdm(range(len(points) - 1)):
            
            origin = points[j, :]
            normal = vectors[j, :]
            min_pressure, avg_pressure, max_pressure = get_pressure(
                data_file,origin, normal, name
            )
            print("   $$ Control point ", j, " : Pressure = ", avg_pressure)
            Lmin.append(min_pressure)
            Lavg.append(avg_pressure)
            Lmax.append(max_pressure)

            Lpressure = np.array([Lmin, Lavg, Lmax])

    else:
        L = [0] * (len(points))
        Lpressure = np.array([L, L, L])

    return Lpressure


def plot_linear(i_vessel, Lpress, dpoints_u, dvectors_u, dist):
    """


    Parameters
    ----------
    i_vessel : index of the vessel.
    Lpress : array of the pressures along the vessel.

    Returns
    -------
    fig : plot of the pressure along the vessel, linear interpolation

    """

    Ldist = []
    

    # remplacer par dpoints et dvectors updated

    # origin = dpoints_u.get("points{}".format(i_vessel))[1]
    # vectors = dvectors_u.get("vectors{}".format(i_vessel))[1]

    for i in range(0, dist.shape[0] - 1):
        Ldist.append(float((dist[i] + dist[i + 1]) / 2))

    fig = plt.figure(figsize=(14.4, 10.8))

    # plt.plot(Ldist, Lpress[0,:], '--',label='Minimum')
    # plt.plot(Ldist, Lpress[0,:], '^', label='Minimum')

    plt.plot(Ldist, Lpress[1, :], "--")
    plt.plot(Ldist, Lpress[1, :], "o", label="Average")

    # plt.plot(Ldist, Lpress[2,:], '--',label='Maximum')
    # plt.plot(Ldist, Lpress[2,:], 'v', label='Maximum')

    plt.fill_between(Ldist, Lpress[0, :], Lpress[2, :], alpha=0.2)

    plt.grid()
    plt.xlabel("distance along the vessel (m)", fontsize=18)
    plt.ylabel("Pressure", fontsize=18)
    plt.title(
        "Pressure along the " + dpoints_u.get("points{}".format(i_vessel))[0], fontsize=20
    )
    plt.legend(loc="best")

    plt.show()

    return fig


def save_pressure(data_file,i, dpoints, dvectors):
    
    """


    Parameters
    ----------
    i : vessel index.

    Returns
    -------
    dpressure : dictionnary of the pressure in the vessel i.

    """
    dpressure = {}
    # dpressure['Informations']=pinfo,case,filename
    Lpress = compute_along(data_file,i, dpoints, dvectors)

    pressure_array = invert_array(np.array(Lpress))
    name = dpoints.get("points{}".format(i))[0]
    dpressure["pressure{}".format(i)] = name, pressure_array

    return dpressure


def save_pressure_all(data_file,dpoints, dvectors):
    """


    Returns
    -------
    dpressure : dictionary of the pressure in all the vessels

    """
    dpressure = {}
    for i in range(len(dpoints)):
        Lpress = compute_along(data_file,i, dpoints, dvectors)
        pressure_array = invert_array(np.array(Lpress))
        name = dpoints.get("points{}".format(i))[0]
        dpressure["pressure{}".format(i)] = name, pressure_array

    return dpressure


def plot_on_time(dpressure, i_vessel, pinfo):

    onlydat, indices = get_list_files_dat(pinfo, case, num_cycle)
    len_vessel = (
        dpressure.get("{}".format(indices[0]))
        .get("pressure{}".format(i_vessel))[1]
        .shape[0]
    )
    name_vessel = dpressure.get("{}".format(indices[0])).get(
        "pressure{}".format(i_vessel)
    )[0]
    tabY = np.zeros((len_vessel))
    X = np.linspace(1, len(onlydat) - 1, len(onlydat) - 1)

    for i in range(len_vessel):

        Y = [
            dpressure.get("{}".format(indices[j])).get("pressure{}".format(i_vessel))[
                1
            ][i, 1]
            for j in range(len(onlydat) - 1)
        ]
        # modify by taking the second line (avg)
        plt.plot(X, Y)
    plt.grid()

    plt.xlabel("Time step")
    plt.ylabel("Pressure")
    plt.title("Pressure in each point in function of time in the " + name_vessel)
    plt.savefig(
        "N:/vasospasm/pressure_pytec_scripts/plots_time/"
        + pinfo
        + "_"
        + case
        + "_"
        + name_vessel
        + ".png"
    )
    plt.show()


def plot_time_dispersion(dpressure, i_vessel, pinfo,case):

    Ldist = []

    onlydat, indices = get_list_files_dat(pinfo, case, num_cycle)
    print(indices)
    len_vessel = (
        dpressure.get("{}".format(indices[0]))
        .get("pressure{}".format(i_vessel))[1][1]
        .shape[1]
    )
    name_vessel = dpressure.get("{}".format(indices[0])).get(
        "pressure{}".format(i_vessel)
    )[0]
    
    variation = get_variation(pinfo, case)

    os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")
    module_name = "division_variation" + str(variation)
    module = importlib.import_module(module_name)

    importlib.reload(module)
    importlib.reload(geom)
    dpoints_u, dvectors_u = module._main_(pinfo, case, step)

   
    
    ddist = {}
    
    ddist = get_distance_along(
            i_vessel, 0, dpressure, dvectors_u, dpoints_u,pinfo,case
        )
    

    dist = ddist.get("dist{}".format(i_vessel))[1]
    for i in range(0, dist.shape[0] - 1):
        Ldist.append(float((dist[i] + dist[i + 1]) / 2))

    tab_pressure = np.zeros((len_vessel, 3))
    len_cycle=30
    for i in range(len_vessel):
       Lmin = [
           dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[1][1][0, i]
            for k in range(len_cycle)
        ]
       
       Lmean = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][1, i]
            for k in range(len_cycle)
        ]
       Lmax = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][2, i]
            for k in range(len_cycle)
        ]

       tab_pressure[i, 0] = sum(Lmin) / len(Lmin)
       tab_pressure[i, 1] = sum(Lmean) / len(Lmean)
       tab_pressure[i, 2] = sum(Lmax) / len(Lmax)

    fig = plt.figure(figsize=(14.4, 10.8))

    plt.plot(Ldist, tab_pressure[:, 1], "--")
    plt.plot(Ldist, tab_pressure[:, 1], "o", label="Average pressure over time")
    plt.fill_between(
        Ldist,
        tab_pressure[:, 0],
        tab_pressure[:, 2],
        alpha=0.2,
        label="average enveloppe of min/max pressure over time",
    )

    plt.grid()
    plt.xlabel("distance along the vessel (m)", fontsize=18)
    plt.ylabel("Pressure", fontsize=18)
    plt.title(
        "Pressure along the " + ddist.get("dist{}".format(i_vessel))[0], fontsize=20
    )
    plt.legend(loc="best")

    plt.savefig(
        "N:/vasospasm/pressure_pytec_scripts/plots_resistance/"
        + pinfo
        + "_"
        + case
        + "_"
        + name_vessel
        + ".png"
    )
    plt.show()

    return fig


def invert_array(arr):
    new_arr = np.ones_like(arr)
    if arr[1, 0] < arr[1, arr.shape[1] - 1]:
        for i in range(3):
            new_arr[i, :] = arr[i][::-1]
        return 1, new_arr
    else:
        return 0, arr


def get_Q_final(pinfo, case):

    dinfo = scipy.io.loadmat(
        "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/case_info.mat"
    )
    dqfinal = dinfo.get("Q_final")
    Q_arr = np.zeros((30, 11))

    for i in range(0, dqfinal.shape[0] // 2 - 3, 3):
        p = i // 3
        Q_arr[p][:] = dqfinal[i][:]

    for i in range(dqfinal.shape[0] // 2 + 9, dqfinal.shape[0], 3):
        p = i // 3 - 3
        Q_arr[p][:] = dqfinal[i][:]

    return Q_arr


def plot_R(dpressure,ddist, i_vessel, pinfo, case, ax):

    Ldist = []

    onlydat, indices = get_list_files_dat(pinfo, case, num_cycle)
    len_vessel = (
        dpressure.get("{}".format(indices[0]))
        .get("pressure{}".format(i_vessel))[1][1]
        .shape[1]
    )
    name_vessel = dpressure.get("{}".format(indices[0])).get(
        "pressure{}".format(i_vessel)
    )[0]
    
    # variation = get_variation(pinfo, case)

    # os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")
    # module_name = "division_variation" + str(variation)
    # module = importlib.import_module(module_name)

    # importlib.reload(module)
    # importlib.reload(geom)
    # dpoints_u, dvectors_u = module._main_(pinfo, case, step)

   
    
    # ddist = {}
    
    # ddist = get_distance_along(
    #         i_vessel, 0, dpressure, dvectors_u, dpoints_u,pinfo,case
    #     )
    # print(ddist)

    dist = ddist.get("dist{}".format(i_vessel))[1]
    for i in range(0, dist.shape[0] - 1):
        Ldist.append(float((dist[i] + dist[i + 1]) / 2))

    tab_pressure = np.zeros((len_vessel, 3))

    Qfinal = np.mean(get_Q_final(pinfo, case)[:, 3])
    len_cycle = 30
    for i in range(len_vessel):
        # print(dpressure.get('{}'.format(indices[0])).get('pressure{}'.format(i_vessel))[1][1][0,i])
        Lmin = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][0, i]
            for k in range(len_cycle)
        ]
        Lmean = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][1, i]
            for k in range(len_cycle)
        ]
        Lmax = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][2, i]
            for k in range(len_cycle)
        ]

        tab_pressure[i, 0] = sum(Lmin) / len(Lmin)
        tab_pressure[i, 1] = sum(Lmean) / len(Lmean)
        tab_pressure[i, 2] = sum(Lmax) / len(Lmax)

    tab_resistance = np.zeros((len_vessel, 3))
    tab_resist_locale = np.zeros((len_vessel - 1, 3))
    for i in range(1, len_vessel):
        tab_resistance[i, :] = (tab_pressure[i, :] - tab_pressure[1, :]) / Qfinal
    for i in range(len_vessel - 1):
        tab_resist_locale[i, :] = (tab_pressure[i , :] - tab_pressure[i+1, :]) / Qfinal

    # fig = plt.figure(figsize=(14.4, 10.8))
    # ax=fig.add_subplot(111)
    # ax.plot(Ldist[:-1], tab_resist_locale[:, 1], label="local resistance")

    #plt.yscale('symlog')

    # plt.plot(Ldist[1:], tab_resistance[1:, 1], "--")
    # plt.plot(
    #     Ldist[1:], tab_resistance[1:, 1], "o", label="Average resistance over time"
    # )
    # plt.fill_between(Ldist,
    #                   tab_resistance[:,
    #                                 0],
    #                   tab_resistance[:,
    #                                 2],
    #                   alpha=0.2,
    #                   label='average enveloppe of min/max resistance over time')

    ax.grid()
    # ax.xlabel("distance along the vessel (m)", fontsize=18)
    # ax.ylabel("Resistance", fontsize=18)
    # ax.title("Resistance along the " + name_vessel, fontsize=20)
    # ax.legend(loc="best")

    # plt.savefig(
    #     "N:/vasospasm/pressure_pytec_scripts/plots_resistance/test/"
    #     + pinfo
    #     + "_"
    #     + case
    #     + "_"
    #     + name_vessel
    #     + ".png"
    # )
    # plt.show()

    return ax.plot(Ldist[:-1], tab_resist_locale[:, 1], label="local resistance | " + case)



# %% Main


def main():
    """

    The object of this script is to compute and plot the pressure along the vessels
    of the circle of Willis of a specific patient for certain points. The methode used is to create slice and extract them into tecplot to get the pressure along.



    This program
        --> Step 1 : Extract files and coordinates, label names. All in a dictionnary

        --> Step 2 : Organize the 'raw' data into clear division of the vessels. Automatically if it is a complete CoW, with some manual input otherwise.
                    The division_variation_{i} scripts make the separation of :
                        - L/_ICA_MCA into L/R_ICA & L/R_MCA
                        - PCA into P1 and P2
                        - BAS_PCA into BAS & P1 & P2
                        - ACA into A1 & A2
                    The script also defines a direction for certain vessels : 
                        For the ACAs, the starting point is at the intersection with the ICA
                        For the PCA, the starting point is at the intersection with the basilar artery.
                    At each bifurcation that occur in the first step, a half radius of the intersecting vessel is remove at each side of the other vessel
                    in the bifurcation. This is to prevent error during the slicing process.

        --> Step 3 : Compute the dictionary of the normal vectors from the new control points

        --> Step 5 : Compute pressure with tecplot

                Step 5.1 : Selecting the good case (patient, case, num_cycle, .dat file), and load data into Tecplot
                Step 5.2 : Find the list of the closest points in fluent to the control points.
                Step 5.3 : Make a slice for each control point and normal vector
                Step 5.4 : Find the subslice that correspond to the vessel in which one is interested
                Step 5.5 : Compute the min, max and average pressure in the subslice
                Step 5.6 : Save the pressures in a dictionnary,containing every cycle/.dat file/ vessel.
                Step 5.7 : change the order of the vessel if necessary to have a decreasing plot of pressure
                
        --> Step 6 : Plot pressure in each vessel

            Step 7 : extract flowrate, compute delta_P = P_{i+1} - P_{i}, and plot time average resistance along the segments.

    """
    
    # Still a few global variables. 

    global pinfo
    global case
    global num_cycle
    global step
    global select_file
    global filename
    global data_file
    
    ######################################### Choosing the patient informations
  
    print("$ Patient informations $\n")
    ptype = input("Patient origin? -- v : vsp## or p: pt## --\n")
    if ptype == "v":
        ptype = "vsp"
    elif ptype == "p":
        ptype = "pt"
    pnumber = input("Patient number?\n")
    pinfo = ptype + pnumber
    case = input("Case ? -- b for baseline of v for vasospasm\n")
    if case == "b":
        case = "baseline"
    elif case == "v":
        case = "vasospasm"

    print("$ Select computing cases\n")
    num_cycle = int(input("Which cycle ? 1,2,3 or 4\n"))

    onlydat, indices_dat = get_list_files_dat(pinfo, case, num_cycle) # Get the dat files for the patient and its case

    for k in range(len(onlydat)):
        print(k, ": " + indices_dat[k][9:-3] + ":" + indices_dat[k][11:] + " s")
    print("a : one period (30 time steps) \n")
    select_file = input("which time step?\n")

    print("$ Step $\n")
    step_in = input()
    step=int(step_in)

    # Load a different module of the control points extraction and sorting depending on the patient variation
    
    ############################################################# STEP 1, 2 & 3
    
    variation = get_variation(pinfo, case) # Get the variation from the case_info.mat

    os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")
    module_name = "division_variation" + str(variation)
    module = importlib.import_module(module_name)

    importlib.reload(module)
    importlib.reload(geom)
    dpoints_u, dvectors_u = module._main_(pinfo, case, step) # Extract the control points, well organized and divided

    
   

    for k in range(len(dpoints_u)):
        print(k, " : " + dpoints_u.get("points{}".format(k))[0])
    print("a : all vessels\n")
    select_vessel = input("Compute on which vessel ?\n")


    dpressure = {}
    dpressure["Informations"] = pinfo, case


    if select_file == "a":
        # Replace by 30 and 60 to compute on the second period 
        start = 0
        end = 30
    else:
        i_file = int(select_file)
        start = i_file
        end = i_file + 1

    for i in range(start, end):
        
        # LOAD THE TECPLOT DATA

        filename = onlydat[i]

        print(
            " ############  Step 2 : Connection to Tecplot  ############ Time step : ",
            i,
            "\n",
        )
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

        print(" ############  Step 3 : Compute pressure  ############\n")

        if select_vessel == "a":
            dpressure["{}".format(indices_dat[i])] = save_pressure_all(
                data_file,dpoints_u, dvectors_u
            )
            dname = "".join(["dpressure", "_", pinfo, "_", case])
            save_dict(dpressure, "N:/vasospasm/pressure_pytec_scripts/" + dname)

        else:
            i_ = int(select_vessel)
            dpress = save_pressure(data_file,i_, dpoints_u, dvectors_u)
            dpressure["{}".format(indices_dat[i])] = dpress
            save_dict(
                dpressure,
                "N:/vasospasm/pressure_pytec_scripts/dpressure" + select_vessel,
            )

        if select_vessel == "a":
            ddist = {}
            for k in range(len(dpoints_u)):
                ddist["{}".format(k)] = get_distance_along(
                    k, i, dpressure, dvectors_u, dpoints_u,pinfo,case
                )

        else:
            i_vessel = int(select_vessel)
            ddist = get_distance_along(i_vessel, i, dpressure, dvectors_u, dpoints_u,pinfo,case)
            save_dict(
                ddist,
                "N:/vasospasm/pressure_pytec_scripts/ddist_" + pinfo + "_" + case ,
            )

        if select_vessel == "a":
            for k in range(len(dpoints_u)):
                Lpress = dpressure.get("{}".format(indices_dat[i])).get(
                    "pressure{}".format(k)
                )[1][1]
                dist = ddist.get("{}".format(k)).get("dist{}".format(k))[1]
                fig = plot_linear(k, Lpress, dpoints_u, dvectors_u, dist)
                fig.savefig(
                    "N:/vasospasm/pressure_pytec_scripts/plots_8_4/"
                    + pinfo
                    + "/"
                    + case
                    + "/cycle_"
                    + str(num_cycle)
                    + "/plot_"
                    + indices_dat[i]
                    + "/"
                    + "plot_"
                    + dpoints_u.get("points{}".format(k))[0]
                    + ".png"
                )

        else:
            i_vessel = int(select_vessel)
            Lpress = dpressure.get("{}".format(indices_dat[i])).get(
                "pressure{}".format(i_)
            )[1][1]
            dist = ddist.get("dist{}".format(i_vessel))[1]
            print("Lpress : ", Lpress)
            print("dist :", dist)
            fig = plot_linear(i_vessel, Lpress, dpoints_u, dvectors_u, dist)
            fig.savefig(
                "N:/vasospasm/pressure_pytec_scripts/plots_8_4/"
                + pinfo
                + "/"
                + case
                + "/cycle_"
                + str(num_cycle)
                + "/plot_"
                + indices_dat[i]
                + "/"
                + "plot_"
                + dpoints_u.get("points{}".format(i_))[0]
                + ".png"
            )

    #plot_R(dpressure,i_vessel,pinfo,case)

    return dpressure,ddist

# if __name__=="__main__":
#     dpressure_rmca_vas,ddist_rmca_vas = main()
    


fig, ax = plt.subplots(1,1)
l1 = plot_R(dpressure_rmca_vas,ddist_rmca_vas,11, 'pt2', 'baseline', ax)
l2 = plot_R(dpress_rmca_vas,ddist_rmca_vas_vas,11, 'pt2', 'vasospasm', ax)
plt.legend()
plt.ylabel('resistance')
plt.xlabel('distance along the vessel (m)')
plt.title('resistance along the RMCA') 
# plt.yscale('log')
plt.grid()
plt.show()

def final_set_of_plots(pinfo,i_vessel,num_cycle,dpoints_bas,dvectors_bas,dpressure_bas,dpoints_vas,dvectors_vas,dpressure_vas):
    
    
    plot_time_dispersion(dpressure_bas,dpressure_vas, i_vessel, pinfo)
    plot_R(dpressure_bas,dpressure_vas, i_vessel, pinfo, cas)
    plot_cross_section(pinfo)
    
    
    
    
    
        
        


