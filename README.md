# Pressure and Resistance determination by slicing - Tecplot project


Python and Pytecplot scripts to compute the pressure and resistance in the vessels of the circle of Willis. The object of this script is to compute and plot the pressure along the vessels 
of the circle of Willis of a specific patient for certain points.

[Add detail]

![plot_slices_lica_pt2b](https://user-images.githubusercontent.com/109392345/182479044-ba92a417-7e28-4e2c-8c7e-d984676c9c31.png)


## Context illustrations

#### Sketch of the anatomy of the circle of Willis

![cercle_willis_arch](https://user-images.githubusercontent.com/109392345/182443114-28fa3a39-a3ca-404c-aeaf-4fab20f74334.png)


#### Segmented Circle of Willis :

<img width="220" alt="circle_of_willis" src="https://user-images.githubusercontent.com/109392345/182442690-a59af0c1-dce3-460e-b07f-73fdb87e9e8f.png">


#### Baseline vs Vasospasm case :


## Architecture of the code

- Step 1 : Extract files and coordinates, label names. All in a dictionnary
- Step 2: Operate a divison for ICA_MCA --> ICA & MCA, PCA --> P1 & P2 and ACA --> A1 & A2.

The division depend of the variation of the circle of Willis of the patient, which are exposed below.
    
### Variations of the circle of Willis :
 | Variation id  | Specification |
| ------------- | ------------- |
| 1  | Complete |
| 2 | Missing Acom  |
| 3  | Missing Left PCA P1  |
| 4 | Missing Right PCA P1  |
| 5  |  Missing Left A1 |
| 6 | Missing Right A1  |
| 7  |  Missing Left Pcom |
| 8 |  Missing Right Pcom  |

For the complete circle of Willis, these division are made automatically with the following vessels to make the separation :
    - ACA for the ICA_MCA
    - Pcom for the PCA
    - Acom for the ACA  
The separation is made finding the closest point of the unsparated vessel from the first/last point of the vessel used to do the separation, by minimazing the euclidian distances.

If there is a vessel missing, the separation is made manually by the user, which enter the coordinates
of the separation point. This is made at the zones impacted by the lack of a vessel in the other variation. Each variation have a python script, and the main() returns 
the complete dictionary of the control points in each vessel.

Hence the dictionary of the normal vectors can be calculated from the dictionary of control points
    
This is just made by substracting the coordinates of the points terms to terms.
- Step 5 : Compute pressure with tecplot
   
   - Step 5.1 : Selecting the good case (patient, case, num_cycle, .dat file), and load data into Tecplot
   - Step 5.2 : Find the list of the closest points in fluent to the control points.
  - Step 5.3 : Make a slice for each control point and normal vector
  - Step 5.4 : Find the subslice that correspond to the vessel in which one is interested
  - Step 5.5 : Compute the average pressure in the subslice
  - Step 5.6 : Save the pressures in a dictionnary,containing every cycle/.dat file/ vessel.
  - Step 5.7 : change the order of the vessel if necessary to have a decreasing plot of pressure
  - Step 6 : Plot pressure in each vessel
        
- Step 7 : extract flowrate, compute delta_P, and plot time average resistance along the segments.
        
## Other features

#### Remove a radius of lenght of the intersecting vessel at a bifurcation :
This is a feature to prevent to slice through two vessels at a time at a bifurcation, and obtain false results

![separation_ica_aca_update](https://user-images.githubusercontent.com/109392345/182442252-4c4e00e2-2c40-4b47-a963-46d48ebea42d.png)

## Results illustrations

#### For patient 2 - baseline vs vasospasm case. Representation of the pressure along the vessel, averaged on time

![pt2_baseline_L_MCA (1)](https://user-images.githubusercontent.com/109392345/182961087-9c1b584c-dee1-48d8-bfd4-c7d40f868c8b.png)

![pt2_vasospasm_L_MCA (1)](https://user-images.githubusercontent.com/109392345/182961116-5e461df0-7ef4-4c81-9dbf-c29e538001a5.png)

#### For patient 2 - baseline vs vasospasm case. Representation of the resistance along the vessel, averaged on time

![pt2_baseline_L_MCA](https://user-images.githubusercontent.com/109392345/182961330-454aafbb-4db5-4976-8a99-73ceb778064a.png)

![pt2_vasospasm_L_MCA_3](https://user-images.githubusercontent.com/109392345/182961356-0d1992da-dc73-4f95-9e05-67c4d198e76e.png)

        
## Useful information

#### Nomenclature
 
For the code to work correctly, there is a unique naming for the vessels. The following table shows the name for left side, juste replace 'L' by 'R' to have the right side names.

| Name of the vessel  | Name of the .pth or .ctgr file |
| ------------- | ------------- |
| Left PCA  | L_PCA |
| Left Basilar&PCA | L_BAS_PCA  |
| Left PCA Pcom (variation 3&4)  | L_PCA_Pcom  |
| Left Pcom | L_Pcom  |
| Left superior cerebellar artery  |  L_sup_cereb |
| Left ACA | L_ACA |
| Left ACA A1  |  L_A1 |
| Left PCA P2 |  L_P2  |

#### Run the code

Start by activating the right python environment :

``` ruby
activate sypder-cf
```



