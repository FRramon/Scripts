# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:20:18 2022

@author: GALADRIEL_GUEST


"""
import os
os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import importlib
import pandas as pd
import seaborn as sns

import main_pressure_project as press_pj
import geometry_slice as geom
import division_variation3 as variation
import test_separation_radius as cross_section
importlib.reload(press_pj)
importlib.reload(geom)
importlib.reload(variation)

#%%


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



#%% LOAD DATA
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
 
#%%

def get_R(dpressure,ddist,dpoints, i_vessel, pinfo, case,num_cycle):

    Ldist = []

    onlydat, indices = get_list_files_dat(pinfo, case, num_cycle)
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
    dptest = {}
    
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
    tab_pressure_norm = np.zeros((len_vessel,1))
    for i in range(len_vessel):
        tab_pressure_norm[i] = tab_pressure[i,1] - tab_pressure[0,1]
        
        
        # tab_pressure[i, 2] = sum(Lmax) / len(Lmax)
    tab_resistance = np.zeros((len_vessel-1, 3))
    tab_resist_locale = np.zeros((len_vessel - 1, 3))
    for i in range(0, len_vessel-1):
        tab_resistance[i, :] = (tab_pressure[0, :] - tab_pressure[i+1, :]) / Qfinal
    for i in range(len_vessel -1):
        tab_resist_locale[i, :] = (tab_pressure[i , :] - tab_pressure[i+1, :]) / Qfinal

    



    return tab_pressure_norm,tab_resistance, Ldist,Qfinal

def find_closest(Ldist_bas,dist_max):
    abs_diff = [abs(Ldist_bas[i]-dist_max) for i in range(len(Ldist_bas))]
    
    dist_opti = abs_diff.index(min(abs_diff))
    # print(dist_opti)
    # print(len(Ldist_bas)-1)
    if dist_opti >=len(Ldist_bas)-1:
        return len(Ldist_bas)-2
    return dist_opti




def get_ultimate_values(dpoints_bas,tab_pressure_bas,tab_pressure_vas,tab_resistance_bas,tab_resistance_vas,Ldist_bas,Ldist_vas,Q_bas,Q_vas,pinfo,ivessel):
    
    ulti_resist = tab_resistance_vas[-1,1]
    ulti_press = tab_pressure_vas[-1]
    dist_max = Ldist_vas[-1]
    
    dist_opti_bas = find_closest(Ldist_bas,dist_max)
    ulti_resist_bas = tab_resistance_bas[dist_opti_bas,1]
    ulti_press_bas = tab_pressure_bas[dist_opti_bas]
    
    percent_resist = 100*(ulti_resist - ulti_resist_bas)/ulti_resist_bas
    percent_press = 100*(ulti_press - ulti_press_bas)/ulti_press_bas
    
    ulti_Q = Q_vas
    ulti_Q_bas = Q_bas
    
    percent_Q = (ulti_Q-ulti_Q_bas)/ulti_Q_bas
    
    # data = {'Vasospasm pressure': ulti_press,
    #     'Baseline pressure': ulti_press_bas,
    #     'Percentage pressure': percent_press,
    #     'Vasospasm resistance': ulti_resist,
    #     'Baseline resistance': ulti_resist_bas,
    #     'Percentage resistance ': percent_resist,

    #     }
    data = {
        'pressure baseline': ulti_press_bas,
        'pressure vasospasm': ulti_press,
        'Final pressure rate': percent_press,
        'flowrate baseline': ulti_Q_bas,
        'flowrate vasospasm': ulti_Q,
        'Final flowrate rate': percent_Q,
        'resistance baseline': ulti_resist_bas,
        'resistance vasospasm': ulti_resist,
        'Final resistance rate': percent_resist,

        }
    
    
    df = pd.DataFrame(data,index = [dpoints_bas.get('points{}'.format(i))[0]])
    df.loc()
    
    return df

def plot_heatmap(pinfo,num_cycle,dpressure_bas,dpressure_vas,ddist_bas,ddist_vas,dpoints_bas,dpoints_vas):
    
    
    Lvessel=['L_MCA','R_MCA','L_A1','L_A2','R_A1','R_A2','L_P1','L_P2','R_P1','R_P2','BAS','L_ICA','R_ICA']
      
    Lvessel_pth=[dpoints_bas.get('points{}'.format(i))[0] for i in range(len(dpoints_bas))]
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
                
    
                
    df = pd.DataFrame()
          
    for k in range(len(L_ind)):
        
        i=L_ind[k]
        len_vessel_bas = dpoints_bas.get('points{}'.format(i))[1].shape[0]
        len_vessel_vas = dpoints_vas.get('points{}'.format(i))[1].shape[0]
        
        if len_vessel_bas > 2 and len_vessel_vas > 2:
        
            tab_pressure_bas, tab_resistance_bas, Ldist_bas, Q_bas = get_R(dpressure_bas,ddist_bas,dpoints_bas,i,pinfo,'baseline',num_cycle)
            tab_pressure_vas,tab_resistance_vas,Ldist_vas,Q_vas = get_R(dpressure_vas,ddist_vas,dpoints_vas,i,pinfo,'vasospasm',num_cycle)
    
            data = get_ultimate_values(dpoints_bas,tab_pressure_bas,tab_pressure_vas,tab_resistance_bas,tab_resistance_vas,Ldist_bas,Ldist_vas,Q_bas,Q_vas,pinfo,i)
        
            df = pd.concat([df,data])
        
    
    
    # ax = sns.heatmap(df,annot=True, cmap =plt.cm.Blues, fmt='.2f',linewidth=0.3, cbar_kws={"shrink": .8})
    
    f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize = (15,10))
    
    plt.suptitle("Final rates for "+ pinfo, fontsize = 20)
    sns.heatmap(df.loc[:,['pressure baseline','pressure vasospasm','Final pressure rate']],annot = True,cmap =plt.cm.Blues,linewidth=0.3,ax=ax1)
    # g1.set_ylabel('')
    # g1.set_xlabel('')
    sns.heatmap(df.loc[:,['flowrate baseline','flowrate vasospasm','Final flowrate rate']],annot = True,cmap =plt.cm.Blues,linewidth=0.3,ax=ax2)
    # g2.set_ylabel('')
    # g2.set_xlabel('')
    ax2.set_yticks([])
    sns.heatmap(df.loc[:,['resistance vasospasm','resistance baseline','Final resistance rate']],annot = True,cmap =plt.cm.Blues,linewidth=0.3,ax=ax3)
    # g3.set_ylabel('')
    # g3.set_xlabel('')
    ax3.set_yticks([])
    
    plt.tight_layout()

#%% 


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
        dCS['slice{}'.format(i)] = cross_section.compute_radius(pinfo,case,num_cycle,step, dpoints, dvectors, i)
    
    return dCS

def plot_CS(dCS,ddist,case):
    for k in range(len(L_ind)):
        i = L_ind[k]
        slice_to_plot = dCS.get('slice{}'.format(i))
        name,dist_forx = ddist.get('dist{}'.format(i))
        print('name vessel : ',name)
        print('nb slice  : ',len(slice_to_plot))
        print('nb dist : ',len(dist_forx))
        len_to_keep = len(slice_to_plot)-len(dist_forx)
        
        print(len_to_keep)
        # adjust sizes
        if len_to_keep>0:
            slice_to_plot = slice_to_plot[:-len_to_keep]
        elif len_to_keep<0:
            dist_forx = dist_forx[:+len_to_keep]
        
        # Clean (remove points to big)
        
        print('\n')
        avg_value = np.mean(slice_to_plot)
        for l in range(len(slice_to_plot)):
            if slice_to_plot[l] > 3 * avg_value:
                if l>= 1:
                    slice_to_plot[l] = slice_to_plot[l-1]
                else : 
                    slice_to_plot[l] = slice_to_plot[3]
        print('name vessel : ',name)
        print('nb slice  : ',len(slice_to_plot))
        print('nb dist : ',len(dist_forx))
        
        plt.plot(dist_forx,slice_to_plot)
        plt.show()
        
    plt.show()
    
    
    
    
    
    
