# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:20:18 2022

@author: GALADRIEL_GUEST
"""
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pickle
import importlib


import main_pressure_project as press_pj
import geometry_slice as geom
import division_variation3 as variation
importlib.reload(press_pj)
importlib.reload(geom)
importlib.reload(variation)


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



def dpress_offset(dpressure_bas,dpressure,ddist,dpoints, i_vessel, pinfo,num_cycle):

    Ldist = []

    case = 'vasospasm' 
    onlydat, indices = get_list_files_dat(pinfo, case, num_cycle)
    onlydat_b , indices_b = get_list_files_dat(pinfo,'vasospasm', num_cycle)

    len_vessel = (
        dpressure.get("{}".format(indices[0]))
        .get("pressure{}".format(i_vessel))[1][1]
        .shape[1]
    )
    name_vessel = dpoints.get("points{}".format(i_vessel))[0]
    
    print(name_vessel)
  
    dist = ddist.get("dist{}".format(i_vessel))[1]
    for i in range(0, dist.shape[0] - 1):
        Ldist.append(float((dist[i] + dist[i + 1]) / 2))
    
    tab_pressure = np.zeros((len_vessel, 3))
    min_diff= []
    max_diff= []
  
    len_cycle = 30
    L_min_env = []
    L_max_env = []
    # FINDING THE TIME STEPS WITH THE MINIMUM VALUE AND THE MAXIMUM VALUE TO CREATE THE ENVELOP
    for  k in range(len_cycle):
        Lmin_onets = np.mean([
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][0, i]
            for i in range(len_vessel)
        ])
        L_min_env.append(Lmin_onets)
        Lmax_onets = np.mean([
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][2, i]
            for i in range(len_vessel)
        ])
        L_max_env.append(Lmax_onets)
    index_min_env=L_min_env.index(min(L_min_env))
    index_max_env=L_max_env.index(max(L_max_env))
    
    
    # Finding the time step with the min variation of pressure
    
    
    tab_enveloppe = np.zeros((2,len_vessel))
    Lmean_baseline = [
        dpressure_bas.get("{}".format(indices_b[k])).get("pressure{}".format(i_vessel))[
            1
        ][1][1, 0]
        for k in range(len_cycle)
    ]
    reference = sum(Lmean_baseline)/len(Lmean_baseline)
    
    
    for i in range(len_vessel):
        # print(dpressure.get('{}'.format(indices[0])).get('pressure{}'.format(i_vessel))[1][1][0,i])
        Lmin = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][0, i] + reference
            for k in range(len_cycle)
        ]
        Lmean = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][1, i] + reference
            for k in range(len_cycle)
        ]
        Lmax = [
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][2, i] +reference
            for k in range(len_cycle)
        ]
        
       
        tab_enveloppe[0]=  [ dpressure.get("{}".format(indices[index_min_env])).get("pressure{}".format(i_vessel))[1][1][0, i]]
        tab_enveloppe[1]=  [ dpressure.get("{}".format(indices[index_max_env])).get("pressure{}".format(i_vessel))[1][1][0, i]]

        #min_diff.append([min(min(Lmin[j]-Lmin[j+1]),min(Lmean[j]-Lmean[j+1]),min(Lmax[j]-Lmax[j+1])) for j in range(len_vessel-1)])
        # max_diff.append([max((Lmin[j]-Lmin[j+1]),(Lmean[j]-Lmean[j+1]),(Lmax[j]-Lmax[j+1])) for j in range(len_vessel-1)])
        # tab_pressure[i, 0] = sum(Lmin) / len(Lmin)
        tab_pressure[i, 1] = sum(Lmean) / len(Lmean)
        # tab_pressure[i, 2] = sum(Lmax) / len(Lmax)

    return tab_enveloppe,tab_pressure



#%%
step = 10
num_cycle = 2
pinfo = 'pt2'
dpoints_bas ,dvectors_bas = variation._main_(pinfo,'baseline',step)
dpoints_vas ,dvectors_vas = variation._main_(pinfo,'vasospasm',step)
 # Replace by load dict & return theese dict in main project
 
 
dpressure_pt2_bas = load_dict('N:/vasospasm/pressure_pytec_scripts/plots_8_4/dpressure_pt2_bas')
ddist_pt2_bas_raw = load_dict('N:/vasospasm/pressure_pytec_scripts/plots_8_4/ddist_pt2_bas')
dpressure_pt2_vas = load_dict('N:/vasospasm/pressure_pytec_scripts/plots_8_4/dpressure_pt2_vas')
ddist_pt2_vas_raw = load_dict('N:/vasospasm/pressure_pytec_scripts/plots_8_4/ddist_pt2_vas')
 
 
ddist_pt2_bas = {}
for i in range(len(ddist_pt2_bas_raw)):
    ddist_pt2_bas['dist{}'.format(i)] = ddist_pt2_bas_raw.get('{}'.format(i)).get('dist{}'.format(i))
 
ddist_pt2_vas = {}
for i in range(len(ddist_pt2_vas_raw)):
    ddist_pt2_vas['dist{}'.format(i)] = ddist_pt2_vas_raw.get('{}'.format(i)).get('dist{}'.format(i))
 
Q2bas,lnames = press_pj.get_Q_final('pt2', 'baseline', dpoints_bas, 2)
 


def plot_R(dpressure,ddist,dpoints, i_vessel, pinfo, case,num_cycle, ax,ax2):

    Ldist = []

    onlydat, indices = get_list_files_dat(pinfo, case, num_cycle)
    len_vessel = (
        dpressure.get("{}".format(indices[0]))
        .get("pressure{}".format(i_vessel))[1][1]
        .shape[1]
    )
    name_vessel = dpoints.get("points{}".format(i_vessel))[0]
    
    print(name_vessel)
  
    dist = ddist.get("dist{}".format(i_vessel))[1]
    for i in range(0, dist.shape[0] - 1):
        Ldist.append(float((dist[i] + dist[i + 1]) / 2))
    
    tab_pressure = np.zeros((len_vessel, 3))
    min_diff= []
    max_diff= []
    dQ,list_name = press_pj.get_Q_final(pinfo, case,dpoints,num_cycle)
    
    Qfinal = np.mean(dQ.get('Q_{}'.format(name_vessel))[:])
    len_cycle = 30
    L_min_env = []
    L_max_env = []
    # FINDING THE TIME STEPS WITH THE MINIMUM VALUE AND THE MAXIMUM VALUE TO CREATE THE ENVELOP
    for  k in range(len_cycle):
        Lmin_onets = np.mean([
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][0, i]
            for i in range(len_vessel)
        ])
        L_min_env.append(Lmin_onets)
        Lmax_onets = np.mean([
            dpressure.get("{}".format(indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][2, i]
            for i in range(len_vessel)
        ])
        L_max_env.append(Lmax_onets)
    index_min_env=L_min_env.index(min(L_min_env))
    index_max_env=L_max_env.index(max(L_max_env))
    
    
    # Finding the time step with the min variation of pressure
    
    
    tab_enveloppe = np.zeros((2,len_vessel))
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
        
       
        tab_enveloppe[0]=  [ dpressure.get("{}".format(indices[index_min_env])).get("pressure{}".format(i_vessel))[1][1][0, i]]
        tab_enveloppe[1]=  [ dpressure.get("{}".format(indices[index_max_env])).get("pressure{}".format(i_vessel))[1][1][0, i]]

        #min_diff.append([min(min(Lmin[j]-Lmin[j+1]),min(Lmean[j]-Lmean[j+1]),min(Lmax[j]-Lmax[j+1])) for j in range(len_vessel-1)])
        # max_diff.append([max((Lmin[j]-Lmin[j+1]),(Lmean[j]-Lmean[j+1]),(Lmax[j]-Lmax[j+1])) for j in range(len_vessel-1)])
        # tab_pressure[i, 0] = sum(Lmin) / len(Lmin)
        tab_pressure[i, 1] = sum(Lmean) / len(Lmean)
        
        dpressure['press{}'.format(i)] = tab_enveloppe[0],tab_pressure[i,1],tab_enveloppe[1]
        # tab_pressure[i, 2] = sum(Lmax) / len(Lmax)
    tab_resistance = np.zeros((len_vessel-1, 3))
    tab_resist_locale = np.zeros((len_vessel - 1, 3))
    for i in range(0, len_vessel-1):
        tab_resistance[i, :] = (tab_pressure[0, :] - tab_pressure[i+1, :]) / Qfinal
    for i in range(len_vessel -1):
        tab_resist_locale[i, :] = (tab_pressure[i , :] - tab_pressure[i+1, :]) / Qfinal

    return dpressure



