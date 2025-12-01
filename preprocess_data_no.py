# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:45:07 2024

@author: liang
"""

import sys
sys.path.append('D:/MLFEA/code/mesh')
import numpy as np
import torch
import time
from PolygonMeshProcessing import TriangleMesh, PolygonMesh, SmoothAndProject, SimpleSmootherForMesh
from LoadMeshFromINPFile import read_abaqus_inp
from PolyhedronMeshProcessing import PolyhedronMesh, TetrahedronMesh, Tet10Mesh, ExtractSurface
from SavePointAsVTKFile import save_point_as_vtk
#%% load inp
filename="D:/MLFEA/minliang_lv/data/ori/p54_c3d4_ori"
out=read_abaqus_inp(filename+".inp", remove_unused_node=True)
node=out['node']
node_set=out['node_set']
element=out['element']
element_type=out['element_type']
element_set=out['element_set']
element_orientation=out['element_orientation']
print('element_orientation', element_orientation.shape, np.abs(element_orientation).sum())
print(element_set.keys())
print(node_set.keys())
#%% 
print("open the vtk file in paraview and select three points, then run the code below")
sys.exit()
#%% re-orient the mesh so that it will be easy to generate mat distribution
# choose three points on the curve of mitral annulus using paraview 
#  y
#  ^
#  |    
#  p1
#  |-----p0 (->x)
#  p2
#------------------------------
#modify these manually
p0=9872; p1=9889; p2=9854
#------------------------------
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
#%%
if 'c3d4' in filename:
    LV_solid=TetrahedronMesh(node, element)
elif 'c3d10' in filename:
    LV_solid=Tet10Mesh(node, element)
else:
    raise ValueError("only support c3d4 and c3d10")    
#%%
surface=ExtractSurface(LV_solid)
surface=TriangleMesh(node, surface)
surface.save_as_vtk(filename+"_surface.vtk")
#%%
surface.update_node_normal()
surface.node=surface.node+surface.node_normal
surface.save_as_vtk(filename+"_surface_offset.vtk")
#%%
Boundary=[]
for key, value in node_set.items():   
    Boundary.extend(value) 



