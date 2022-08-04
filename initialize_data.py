# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:03:03 2022

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
from tqdm import trange

os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

import geometry_slice as geom




#%%


pinfo='pt7'
case='baseline'
step=15





def compute_criteria(dvectors):
    dnorms={}
    for i in range(len(dvectors)):
        array_vectors=dvectors.get('vectors{}'.format(i))[1]
        array_norms=np.zeros((array_vectors.shape[0]))
        for j in range(array_vectors.shape[0]):
            array_norms[j]=np.linalg.norm(array_vectors[j])
            
        dnorms['norms{}'.format(i)]=array_norms
        
    L_avg=[]
    for i in range(len(dnorms)):
        L_avg.append(np.mean(dnorms.get('norms{}'.format(i))))
        
    
    L_std=[]
    for i in range(len(dnorms)):
        L_std.append(np.std(dnorms.get('norms{}'.format(i))))
        
    criteria= np.mean(L_avg) + 3*np.mean(L_std)
        
    return criteria



pinfo='pt7'
case='baseline'



f = open('N:/vasospasm/' + pinfo + '/' + case+ '/1-geometry/' + filename)
data = np.loadtxt(f,
                  dtype=str,
                  usecols=(0,1,2),
                  delimiter=' ')
L_coord=[float(x) for x in data[1,:]]
sep_point=np.array(L_coord)


def crop_ICAs(dpoints):
    
      
    f = open('N:/vasospasm/' + pinfo + '/' + case+ '/1-geometry/' + filename)
    data = np.genfromtxt(f,
                     skip_header=0,
                     skip_footer=0,
                     names=True,
                     dtype=None,
                     usecols=(0,1,2),
                     delimiter=' ')

    
    importlib.reload(geom)

   

# sed = np. load('N:/vasospasm/' + pinfo + '/' + case+ '/1-geometry/' + filename)
# f = open('N:/vasospasm/' + pinfo + '/' + case+ '/1-geometry/' + filename)
# class1 = pickle.load(f)
   
    
    
    if side=='L':
        points_ica=dpoints.get('points{}'.format(9))[1]
        sep_point=np.array([-0.003168023657053709, -0.03868599608540535, -0.4955081045627594])
    elif side=='R':
        points_ica=dpoints.get('points{}'.format(10))[1]
        sep_point=np.array([-0.024270353838801384, -0.036143455654382706, -0.491994708776474])
    
    points_ant,points_sup=geom.bifurcation_one(points_ica,sep_point)
    
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid()
    
    ax.scatter(
        points_ant[:,0],points_ant[:,1],points_ant[:,2],label='ant'
        )
    ax.scatter(
        points_sup[:,0],points_sup[:,1],points_sup[:,2],label='sup'
        )
    plt.legend()
    ax.view_init(30,90)
    plt.show()

    return points_sup



#%%Test




def initialize_data():
    
      
    pinfo='pt7'
    case='baseline'
    step=10

    
    os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

    module_name = "division_variation1"
    module = importlib.import_module(module_name)


    importlib.reload(module)
    importlib.reload(geom)
    dpoints, dvectors = module._main_(pinfo, case, step)
        
  
    
    print(
        ' ############  Step 1 : First Connection to Tecplot  ############ Initializing data : Final Geometry and distances along the vessels \n')

    logging.basicConfig(level=logging.DEBUG)

    # Run this script with "-c" to connect to Tecplot 360 on port 7600
    # To enable connections in Tecplot 360, click on:
    #   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

    tp.session.connect()
    tp.new_layout()
    frame = tp.active_frame()

    dir_file = 'N:/vasospasm/' + pinfo + '/' + case+ \
        '/1-geometry/' + filename

    data_file_baseline = tp.data.load_fluent(
        case_filenames=[
            'N:/vasospasm/' +
            pinfo +
            '/' +
            case+
            '/3-computational/hyak_submit/' +
            pinfo +
            '_' +
            case+
            '.cas'],
        data_filenames=[dir_file])
    
    name= pinfo + '_' + case + '.walls'
    
    cx = data_file_baseline.zone(name).values("X")[:]
    cy = data_file_baseline.zone(name).values("Y")[:]
    cz = data_file_baseline.zone(name).values("Z")[:]
    x_base = np.asarray(cx)
    y_base = np.asarray(cy)
    z_base = np.asarray(cz)
    
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid()

    
    
    ax.scatter(
           x_base,y_base,z_base
        )
    plt.show()

    



    coordinates_to_keep = np.array([x_base, y_base, z_base]).T
    criteria_norm=compute_criteria(dvectors)
    dpoints_adapt_model={}
    
    
    for i in trange(9,10):
        points_vessel=dpoints.get('points{}'.format(i))[1]
        for j in trange(points_vessel.shape[0],position=0):
            min_norm=np.min([np.linalg.norm(points_vessel[j]-x) for x in coordinates_to_keep])
            print(min_norm)
            print(criteria_norm)
            if min_norm > criteria_norm:
                points_vessel[j]=np.zeros((1,3))
        num_zero=0
        L_index_z=[]
        for j in range(points_vessel.shape[0]):
            if np.array_equal(points_vessel[i],np.array([0,0,0])):
                num_zero+=1
                L_index_z.append(j)
                
        new_points_vessel=np.zeros((points_vessel.shape[0]-num_zero,3))
        
        L_index_non_z=[i for i in range(points_vessel.shape[0]) if i not in L_index_z]
        k=0
        for j in L_index_non_z:
            new_points_vessel[k]=points_vessel[j]
            k+=1
            
        dpoints_adapt_model['points{}'.format(i)]=new_points_vessel
        
    return dpoints_adapt_model


# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# ax.grid()


# for i in range(len(dlica)):
#     points = dpoints_adapt.get('points{}'.format(i))
    
#     ax.scatter(
#         points[:, 0], points[:, 1], points[:, 2]
#         )

    
# plt.show()
# os.chdir("N:/vasospasm/pressure_pytec_scripts/Scripts")

# module_name = "division_variation1"
# module = importlib.import_module(module_name)


# importlib.reload(module)
# importlib.reload(geom)
# dpoints_u, dvectors_u = module._main_(pinfo, case, step)
         