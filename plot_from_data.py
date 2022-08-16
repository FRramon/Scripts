# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:47:24 2022

@author: GALADRIEL_GUEST
"""

import matplotlib.pyplot as plt
import importlib
import pickle
import numpy as np

import main_pressure_project as press_pj
import geometry_slice as geom
import division_variation3 as variation
importlib.reload(press_pj)
importlib.reload(geom)
importlib.reload(variation)
#Load etc


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

def main(pinfo,step,num_cycle):
    
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
    
        
    Lvessel=['L_MCA','R_MCA','L_A1','L_A2','R_A1','R_A2','L_P1','L_P2','R_P1','R_P2','BAS','L_ICA','R_ICA']
    
      # ind=Lvessel.index(name_vessel)
      
    Lvessel_pth=[dpoints_bas.get('points{}'.format(i))[0] for i in range(len(dpoints_bas))]
    Lvessel_comp=Lvessel_pth.copy()
    
    
    # for x in Lvessel_comp:
    #     if 'L_P1' in x:
    #         Lvessel_comp[Lvessel_comp.index(x)] = 'L_PCA'
    #     if 'L_P2' in x:
    #         Lvessel_comp[Lvessel_comp.index(x)] = 'L_PCA'
    #     if 'R_P1' in x:
    #         Lvessel_comp[Lvessel_comp.index(x)] = 'R_PCA'
    #     if 'R_P2' in x:
    #         Lvessel_comp[Lvessel_comp.index(x)] = 'R_PCA'
    #     if 'R_A2' in x:
    #         Lvessel_comp[Lvessel_comp.index(x)] = 'R_ACA'
    #     if 'R_A1' in x:
    #         Lvessel_comp[Lvessel_comp.index(x)] = 'R_ACA'
    #     if 'L_A2' in x:
    #         Lvessel_comp[Lvessel_comp.index(x)] = 'L_ACA'
    #     if 'L_A1' in x:
    #         Lvessel_comp[Lvessel_comp.index(x)] = 'L_ACA'
    
    
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
                
    
                
            
    for k in range(len(L_ind)):
        
        i=L_ind[k]
        print(i)
        # print(i)
        #fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        fig,(ax1,ax4,ax2,ax3)=plt.subplots(4,1,figsize=(10,15))
        plt.rcParams['axes.grid'] = True

        plt.suptitle("Plots in the " + dpoints_bas.get('points{}'.format(i))[0])
        
        l1 = press_pj.plot_R(dpressure_pt2_bas,ddist_pt2_bas,dpoints_bas,i, pinfo, 'baseline',num_cycle, ax1,ax4)
        l2 = press_pj.plot_R(dpressure_pt2_vas,ddist_pt2_vas,dpoints_vas, i , pinfo, 'vasospasm',num_cycle, ax1,ax4)
        ax1.set_ylabel('resistance')
        ax1.set_title('local resistance along the vessel',fontsize='small' )
        ax1.legend(fontsize='small')
        # plt.grid()
        ax4.set_xlabel('distance along the vessel (m)',fontsize='small')
        ax4.set_ylabel('resistance')
        ax4.set_title('global resistance along the vessel',fontsize='small' )
    
        ax4.legend(fontsize='small')
        # plt.grid()
        
        # plt.show()
        # plt.yscale('log')
        
        
        
        
        l1 = press_pj.plot_time_dispersion(dpressure_pt2_bas,ddist_pt2_bas,i, pinfo, 'baseline',num_cycle, ax2)
        l2 = press_pj.plot_time_dispersion(dpressure_pt2_vas,ddist_pt2_vas,i, pinfo, 'vasospasm',num_cycle, ax2)
        ax2.set_ylabel('pressure')
        ax2.set_xlabel('distance along the vessel (m)',fontsize='small')
        ax2.set_title('pressure along the vessel',fontsize='small') 
    #     #plt.yscale('log')
        ax2.legend(fontsize='small')
    #     plt.grid()
    #     # plt.show()
        
        
        # fig, ax3 = plt.subplots(1,1)
        # l1 = press_pj.plot_cross_section('pt2', 'R_ICA_MCA', ax3)
        # ax3.set_ylabel('radius (m)')
        # ax3.set_xlabel('Distance along the vessel')
        # ax3.set_title("Cross section along the R_ICA_MCA",fontsize='small' )
        # #plt.yscale('log')
        # ax3.legend(fontsize='small')
        
        fig.tight_layout()
        

        plt.show()
    
    
if __name__ == '__main__':
    main('pt2',10,2)
        
        
        
        # def final_set_of_plots(pinfo,i_vessel,num_cycle,dpoints_bas,dvectors_bas,dpressure_bas,dpoints_vas,dvectors_vas,dpressure_vas):
        
        
    #     plot_time_dispersion(dpressure_bas,dpressure_vas, i_vessel, pinfo)
    #     plot_R(dpressure_bas,dpressure_vas, i_vessel, pinfo, cas)
    #     plot_cross_section(pinfo)
        
    #fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    
    
    # fig,(ax1,ax4,ax2,ax3)=plt.subplots(4,1,figsize=(10,15))
    
    # l1 = press_pj.plot_R(dpressure_rmca_vas,ddist_rmca_vas,11, 'pt2', 'baseline',2, ax1,ax4)
    # l2 = press_pj.plot_R(dpress_rmca_vas,ddist_rmca_vas_vas,11, 'pt2', 'vasospasm',2, ax1,ax4)
    # ax1.set_ylabel('resistance')
    # ax1.set_title('resistance along the RMCA',fontsize='small' )
    # ax1.legend(fontsize='small')
    # plt.grid()
    # ax4.set_xlabel('distance along the vessel (m)',fontsize='small')
    # ax4.set_ylabel('resistance')
    
    # ax4.legend(fontsize='small')
    # plt.grid()
    
    # # plt.show()
    # # plt.yscale('log')
    
    
    
    # # fig,ax2=plt.subplots(1,1)
    
    # l1 = press_pj.plot_time_dispersion(dpressure_rmca_vas,ddist_rmca_vas,11, 'pt2', 'baseline',2, ax2)
    # l2 = press_pj.plot_time_dispersion(dpress_rmca_vas,ddist_rmca_vas_vas,11, 'pt2', 'vasospasm',2, ax2)
    # ax2.set_ylabel('pressure')
    # ax2.set_xlabel('distance along the vessel (m)',fontsize='small')
    # ax2.set_title('pressure along the RMCA',fontsize='small') 
    # #plt.yscale('log')
    # ax2.legend(fontsize='small')
    # plt.grid()
    # # plt.show()
    
    
    # # fig, ax3 = plt.subplots(1,1)
    # l1 = press_pj.plot_cross_section('pt2', 'R_ICA_MCA', ax3)
    # ax3.set_ylabel('radius (m)')
    # ax3.set_xlabel('Distance along the vessel')
    # ax3.set_title("Cross section along the R_ICA_MCA",fontsize='small' )
    # #plt.yscale('log')
    # ax3.legend(fontsize='small')
    
    # fig.tight_layout()
    
    
    # plt.show()
    
    
        
        
    
    
        