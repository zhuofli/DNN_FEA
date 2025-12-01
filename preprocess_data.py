# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:01:47 2024

@author: liang
"""
import sys
sys.path.append("C:/Users/zhuofli/OneDrive - Texas Tech University/Data_Zhuofan/NNFEA/code_original/code/mesh")
import numpy as np
import torch
import time
from PolygonMeshProcessing import TriangleMesh, PolygonMesh, SmoothAndProject, SimpleSmootherForMesh
from LoadMeshFromINPFile import read_abaqus_inp
from PolyhedronMeshProcessing import PolyhedronMesh, TetrahedronMesh, Tet10Mesh, ExtractSurface
from SavePointAsVTKFile import save_point_as_vtk
#%% load inp
filename="C:/Users/zhuofli/OneDrive - Texas Tech University/Data_Zhuofan/NNFEA/data_1119/x2_c3d10_rbori"
out=read_abaqus_inp(filename+".inp", remove_unused_node=True)
node=out['node']
node_set=out['node_set']
element=out['element']
element_type=out['element_type']
element_set=out['element_set']
element_orientation=out['element_orientation']
print('element_orientation', element_orientation.shape)
print(element_set.keys())
print(node_set.keys())
#%% save vtk
if 'c3d4' in filename:
    TetMesh=TetrahedronMesh
elif 'c3d10' in filename:
    TetMesh=Tet10Mesh
else:
    raise ValueError("only support c3d4 and c3d10")
LV_solid=TetMesh(node, element)
LV_solid.save_as_vtk(filename+".vtk")    
print('saved', filename)
print("open the vtk file in paraview and select three points")
sys.exit()
#%% re-orient the mesh so that it will be easy to generate mat distribution
# choose three points (p0, p1, p2) on the curve of mitral annulus using paraview 
#  y
#  ^
#  |    
#  p1
#  |-----p0 (->x)
#  p2
#------------------------------
#modify these manually
p0=9873; p1=428; p2=9854
#------------------------------
if 1:
    print('re-orient the mesh~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    origin=(node[p1]+node[p2])/2
    directionX=node[p0]-origin
    directionY=node[p1]-node[p2]
    directionZ=np.cross(directionX, directionY)
    directionX/=np.linalg.norm(directionX, ord=2)
    directionY/=np.linalg.norm(directionY, ord=2)
    directionZ/=np.linalg.norm(directionZ, ord=2)
    node=node-origin.reshape(1,3)
    node_x=(node*directionX.reshape(1,3)).sum(axis=1, keepdims=True)
    node_y=(node*directionY.reshape(1,3)).sum(axis=1, keepdims=True)
    node_z=(node*directionZ.reshape(1,3)).sum(axis=1, keepdims=True)
    node=np.concatenate([node_x, node_y, node_z], axis=1)
    #update element_orientation
    for k in range(len(element)):
        ori=element_orientation[k]
        d0=np.array([(ori[:,0]*directionX).sum(), (ori[:,0]*directionY).sum(), (ori[:,0]*directionZ).sum()])
        d1=np.array([(ori[:,1]*directionX).sum(), (ori[:,1]*directionY).sum(), (ori[:,1]*directionZ).sum()])
        d2=np.cross(d0, d1)
        #update d1
        d1=np.cross(d2, d0)
        d0_norm=np.linalg.norm(d0, ord=2)
        d1_norm=np.linalg.norm(d1, ord=2)
        d2_norm=np.linalg.norm(d2, ord=2)
        if d0_norm < 1e-8:
            print(k, 'd0_norm', d0_norm)
            break
        if d1_norm < 1e-8:
            print(k, 'd1_norm', d1_norm)
            break
        if d2_norm < 1e-8:
            print(k, 'd2_norm', d2_norm)
            break
        d0=d0/d0_norm
        d1=d1/d1_norm
        d2=d2/d2_norm           
        element_orientation[k,:,0]=d0
        element_orientation[k,:,1]=d1
        element_orientation[k,:,2]=d2    
#%% extract BC
#read inp and find the surf names
#*Dsload
#SURF-1, P, 10.
#*Surface, type=ELEMENT, name=SURF-1
#_SURF-1_S1_1, S1
#_SURF-1_S2_1, S2
#_SURF-1_S4_1, S4
#_SURF-1_S3_1, S3
#---------------------------------------------
surface_name_S1='_SURF-1_S1_1'
surface_name_S2='_SURF-1_S2_1'
surface_name_S3='_SURF-1_S3_1'
surface_name_S4='_SURF-1_S4_1'
#---------------------------------------------
Element_surface_pressure_tri3=[]
Element_surface_pressure_tri6=[]
Element_surface_pressure_poly=[]
for e_idx in element_set[surface_name_S1]:
    elm=element[e_idx]
    if element_type[e_idx] == 'C3D4': #face1: 1,2,3
        Element_surface_pressure_tri3.append([elm[0], elm[1], elm[2]])
    elif element_type[e_idx] == 'C3D10': #face1: 1,5,2,6,3,7
        Element_surface_pressure_tri3.append([elm[0], elm[1], elm[2]])
        Element_surface_pressure_tri6.append([elm[0], elm[1], elm[2], elm[4], elm[5], elm[6]])
        Element_surface_pressure_poly.append([elm[0], elm[4], elm[1], elm[5], elm[2], elm[6]])
for e_idx in element_set[surface_name_S2]:
    elm=element[e_idx]
    if element_type[e_idx] == 'C3D4': #face2: 1,4,2 
        Element_surface_pressure_tri3.append([elm[0], elm[3], elm[1]])
    elif element_type[e_idx] == 'C3D10': #face2: 1,8,4,9,2,5
        Element_surface_pressure_tri3.append([elm[0], elm[3], elm[1]])
        Element_surface_pressure_tri6.append([elm[0], elm[3], elm[1], elm[7], elm[8], elm[4]])
        Element_surface_pressure_poly.append([elm[0], elm[7], elm[3], elm[8], elm[1], elm[4]])
for e_idx in element_set[surface_name_S3]:
    elm=element[e_idx]
    if element_type[e_idx] == 'C3D4': #face3: 2,4,3
        Element_surface_pressure_tri3.append([elm[1], elm[3], elm[2]])
    elif element_type[e_idx] == 'C3D10': #face3: 2,9,4,10,3,6
        Element_surface_pressure_tri3.append([elm[1], elm[3], elm[2]])    
        Element_surface_pressure_tri6.append([elm[1], elm[3], elm[2], elm[8], elm[9], elm[5]])        
        Element_surface_pressure_poly.append([elm[1], elm[8], elm[3], elm[9], elm[2], elm[5]])        
for e_idx in element_set[surface_name_S4]:
    elm=element[e_idx]
    if element_type[e_idx] == 'C3D4': #face4: 1,3,4
        Element_surface_pressure_tri3.append([elm[0], elm[2], elm[3]])
    elif element_type[e_idx] == 'C3D10': #face4: 1,7,3,10,4,8
        Element_surface_pressure_tri3.append([elm[0], elm[2], elm[3]])   
        Element_surface_pressure_tri6.append([elm[0], elm[2], elm[3], elm[6], elm[9], elm[7]])
        Element_surface_pressure_poly.append([elm[0], elm[6], elm[2], elm[9], elm[3], elm[7]])
if element_type[0] == 'C3D4':
    Element_surface_pressure=Element_surface_pressure_tri3
    Element_surface_pressure_poly=Element_surface_pressure_tri3
else:
    Element_surface_pressure=Element_surface_pressure_tri6
Boundary=[]
for key, value in node_set.items():   
    Boundary.extend(value) 
#%% save LV_solid with updated node and more info
LV_solid=TetMesh(node, element)
LV_solid.element_data['orientation']=torch.tensor(element_orientation).reshape(-1,9)
LV_solid.mesh_data['Element_surface_pressure']=Element_surface_pressure
LV_solid.mesh_data['Boundary']=Boundary
LV_solid.mesh_data['p0_p1_p2']=[p0, p1, p2]
LV_solid.save_as_vtk(filename+".vtk")
LV_solid.save_as_torch(filename+".pt")
print('saved', filename)
# check surface (normal) and boundary in paraview
#%%
LV_inner_surface=PolygonMesh(node, Element_surface_pressure_poly)
LV_inner_surface.save_as_vtk(filename+"_inner_surface_poly.vtk")
LV_inner_surface_tri3=TriangleMesh(node, Element_surface_pressure_tri3)
LV_inner_surface_tri3.save_as_vtk(filename+"_inner_surface_tri3.vtk")
LV_inner_surface_tri3.update_node_normal()
LV_inner_surface_tri3.node=LV_inner_surface_tri3.node+LV_inner_surface_tri3.node_normal
LV_inner_surface_tri3.save_as_vtk(filename+"_inner_surface_tri3_offset.vtk")
save_point_as_vtk(node[Boundary], filename+"_Boundary.vtk")
Surface=ExtractSurface(LV_solid)
surface_mesh=PolygonMesh(LV_solid.node, Surface)
surface_mesh.save_as_vtk(filename+"_ExtractSurface.vtk")