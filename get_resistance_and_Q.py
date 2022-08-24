# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:02:11 2022

@author: GALADRIEL_GUEST
"""
import os
import numpy as np
import scipy
import importlib
import glob
import matplotlib.pyplot as plt



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

    pathwd = "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit/"
    # mesh_size = "5"
    # os.chdir(pathwd)
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





def get_Q_final(pinfo, case,dpoints,num_cycle):
    os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")
    FindQ = importlib.import_module('find_non_outlet_Q')
    importlib.reload(FindQ)
    
    dQ = {}

    dinfo = scipy.io.loadmat(
        "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/case_info.mat"
    )
    Lvessel=['L_MCA','R_MCA','L_ACA','R_ACA','L_PCA','R_PCA','BAS','L_ICA','R_ICA']
    # ind=Lvessel.index(name_vessel)
    Lvessel_pth=[dpoints.get('points{}'.format(i))[0] for i in range(len(dpoints))]

    # print(Lvessel,Lvessel_pth)

    Lvessel_comp=Lvessel_pth.copy()

    L_indexes=[int(x) for x in np.linspace(0,100,30)]
    dqfinal = dinfo.get("Q_final")
    Q_arr = np.zeros((30, 11))
    for i in range(dqfinal.shape[1]):
        for k in range(30):
            Q_arr[k,i]=dqfinal[L_indexes[k],i]
            
            
    for x in Lvessel_comp:
        # if 'L_P1' in x:
        #     Lvessel_comp[Lvessel_comp.index(x)] = 'L_PCA'
        if 'L_P2' in x:
            Lvessel_comp[Lvessel_comp.index(x)] = 'L_PCA'
        # if 'R_P1' in x:
        #     Lvessel_comp[Lvessel_comp.index(x)] = 'R_PCA'
        if 'R_P2' in x:
            Lvessel_comp[Lvessel_comp.index(x)] = 'R_PCA'
        if 'R_A2' in x:
            Lvessel_comp[Lvessel_comp.index(x)] = 'R_ACA'
        # if 'R_A1' in x:
        #     Lvessel_comp[Lvessel_comp.index(x)] = 'R_ACA'
        if 'L_A2' in x:
            Lvessel_comp[Lvessel_comp.index(x)] = 'L_ACA'
        # if 'L_A1' in x:
        #     Lvessel_comp[Lvessel_comp.index(x)] = 'L_ACA'
        if 'L_BAS' in x:
            Lvessel_comp[Lvessel_comp.index(x)] = 'BAS'
        if 'R_BAS' in x:
            Lvessel_comp[Lvessel_comp.index(x)] = 'BAS'
                

    Verity = np.zeros((len(Lvessel),len(Lvessel_comp)))

    for i in range(len(Lvessel)):
        for j in range(len(Lvessel_comp)):
            Verity[i,j] = (Lvessel[i] in Lvessel_comp[j]) # verity matrix
    
    L_test = []
    L_ind = []
    for i in range(len(Lvessel)):
        for j in range(len(Lvessel_comp)):
            if Verity[i,j] == 1:
                L_test.append(i)
                L_ind.append(j)
    for k in range(len(L_ind)):
        
        L_to_extract=Verity[:,L_ind[k]]
        
        # print('indice L_ind  : ', L_ind[k])
        for i in range(len(Lvessel)):
            
            if L_to_extract[i] == 1:
                ind = Lvessel.index(Lvessel_comp[L_ind[k]])
                dQ['Q_{}'.format(Lvessel_pth[L_ind[k]])] = Q_arr[:,ind]
                
    dQ_other = FindQ.load_df(pinfo, case, num_cycle)
    dQ.update(dQ_other)
    
    return dQ, list(dQ.keys())
   



def plot_R(dpressure,ddist,dpoints, i_vessel, pinfo, case,num_cycle, ax,ax2):

    Ldist = []

    onlydat, indices,pathwd = get_list_files_dat(pinfo, case, num_cycle)
    len_vessel = (
        dpressure.get("{}".format(indices[0]))
        .get("pressure{}".format(i_vessel))[1][1]
        .shape[1]
    )
    name_vessel = dpoints.get("points{}".format(i_vessel))[0]
    
    # print(name_vessel)
  
    dist = ddist.get("dist{}".format(i_vessel))[1]
    for i in range(0, dist.shape[0] - 1):
        Ldist.append(float((dist[i] + dist[i + 1]) / 2))
    
    tab_pressure = np.zeros((len_vessel, 3))
    min_diff= []
    max_diff= []
    dQ,list_name = get_Q_final(pinfo, case,dpoints,num_cycle)
    
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
        # tab_pressure[i, 2] = sum(Lmax) / len(Lmax)
    tab_resistance = np.zeros((len_vessel-1, 3))
    tab_resist_locale = np.zeros((len_vessel - 1, 3))
    for i in range(0, len_vessel-1):
        tab_resistance[i, :] = (tab_pressure[0, :] - tab_pressure[i+1, :]) / Qfinal
    for i in range(len_vessel -1):
        tab_resist_locale[i, :] = (tab_pressure[i , :] - tab_pressure[i+1, :]) / Qfinal

 
    ax.plot(Ldist[:-1], tab_resist_locale[:, 1], label="Local resistance | " + case )

    ax.fill_between(Ldist[:-1],
                      tab_resist_locale[:,2],
                      tab_resist_locale[:,0],
                      alpha=0.2
                      )
    ax2.fill_between(Ldist[1:],
                      tab_resistance[:,
                                    0],
                      tab_resistance[:,
                                    2],
                      alpha=0.2
                      )

    ax2.plot(
    Ldist[1:], tab_resistance[:, 1], "-", label="Global resistance | " + case
 )
    plt.grid()
    return tab_resistance,ax


