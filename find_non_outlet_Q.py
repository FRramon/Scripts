# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:07:15 2022

@author: GALADRIEL_GUEST
"""

#numpy.loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, quotechar=None, like=None)[source]
import numpy as np
import csv
import os
import glob
import pandas as pd

# np.loadata("N:/vasospasm/pt2/baseline/3-computational/hyak_submit/lp1.out")



def get_list_files_dat2(pinfo, case, num_cycle):
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
    print('bien cette fonction')
    pathwd = "N:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit"
    mesh_size = "5"
    list_files_dir = os.listdir("N:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit")
    for files_dir in list_files_dir:
        if "mesh_size_" + mesh_size in files_dir:
            pathwd = pathwd + "/" + files_dir
    os.chdir(pathwd)
    onlyfiles = []
    print(os.getcwd())
    for file in glob.glob("*.dat"):
        if pinfo + "_" + case + "_cycle" + num_cycle in file : 
            
            onlyfiles.append(file)
    indices = [l[13:-4] for l in onlyfiles]


    return onlyfiles, indices,pathwd



def get_1_out(pinfo,case,num_cycle):
    
    onlyfiles,indices,pathwd = get_list_files_dat2(pinfo,case, num_cycle)
    time_limit = [float(x[9:])/1000 for x in indices[:-31]][-1]
    
    dQ = {}
    
    names = ['la1','lp1','ra1','rp1']
    for name in glob.glob('*.out'):
        name_trunc = name[:-4]
        if name_trunc in names:
            filename = pathwd + '/' + name_trunc + '.out'
    
    
                    
            L_Q = []
            with open(filename, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=" ")
                 # the below statement will skip the first row
                next(csv_reader)
                next(csv_reader)
                next(csv_reader)
                i = 0
                for lines in csv_reader:
                    if float(lines[0])/1000 < time_limit:
                        L_Q.append(float(lines[1]))
            for i in range(len(L_Q)):
                if type(L_Q[i]) != float:
                    print(type(L_Q[i]))
            # print(L_Q)
            L_truncate = [int(x) for x in np.linspace(0,len(L_Q)-1,30)]
            # print(L_truncate)
            Q_final = [L_Q[L_truncate[i]] for i in range(len(L_truncate))]
            final_name = (name_trunc[0] + '_' + name_trunc[1:]).upper()
            dQ['Q_{}'.format(final_name)] = Q_final
    return dQ


def load_df(pinfo,case,num_cycle):
    
    onlyfiles,indices,pathwd = get_list_files_dat2(pinfo,case, num_cycle)
    print(indices)
    timl = [float(x.split('-')[-1])/1000 for x in indices[:-31]][-1]

    os.chdir(pathwd)
    print(os.getcwd())
    dQ = {}
    
    names = ['la1','lp1','ra1','rp1']
    for name in glob.glob('*.out'):
        name_trunc = name[:-4]
        if name_trunc in names:
            #filename = 'N:/vasospasm/' + pinfo + '/' + case + '/3-computational/hyak_submit/' +
            filename = name_trunc + '.out'
            L_Q =[]
            df = pd.read_csv(filename,skiprows = 3,sep = ' ',names = ['Time_step', 'Flow','Velmax','flow_time'])

            timestep = df['flow_time'].values.tolist()
            
            for i in range(len(timestep)):
                if  timestep[i]< timl:
                    L_Q.append(df.at[i,'Flow'])
            for i in range(len(L_Q)):
                if type(L_Q[i]) != float:
                    print(type(L_Q[i]))
            L_truncate = [int(x) for x in np.linspace(0,len(L_Q)-1,30)]
            Q_final = [L_Q[L_truncate[i]] for i in range(len(L_truncate))]
            final_name = (name_trunc[0] + '_' + name_trunc[1:]).upper()
            dQ['Q_{}'.format(final_name)] = np.array(Q_final)
    return dQ
        
                    
                    