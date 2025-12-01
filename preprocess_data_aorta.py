# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:40:13 2024

@author: liang
"""
import sys
sys.path.append('D:/MLFEA/code/mesh')
import numpy as np
import torch
import time
from PolygonMeshProcessing import PolygonMesh, SmoothAndProject, SimpleSmootherForMesh
from LoadMeshFromINPFile import read_abaqus_inp
from PolyhedronMeshProcessing import PolyhedronMesh, TetrahedronMesh, Tet10Mesh, ExtractSurface
from SavePointAsVTKFile import save_point_as_vtk
from SaveMeshAsINPFile import save_polyhedron_mesh_to_inp
#%% load pt
filename='D:/MLFEA/minliang_lv/data/ssm/p0_171_solid_tet4'
aorta_tet4=PolyhedronMesh()
aorta_tet4.load_from_torch(filename+".pt")
aorta_tet4.mesh_data['Element_surface_pressure']=aorta_tet4.element_set['Element_surface_pressure']
Boundary=aorta_tet4.node_set['boundary0'].tolist()+aorta_tet4.node_set['boundary1'].tolist()
aorta_tet4.mesh_data['Boundary']=torch.tensor(Boundary, dtype=torch.int64)
aorta_tet4.save_as_torch(filename+".pt")
#%%
filename='D:/MLFEA/minliang_lv/data/ssm/p0_171_solid_tet10'
aorta_tet10=PolyhedronMesh()
aorta_tet10.load_from_torch(filename+".pt")
aorta_tet10.mesh_data['Element_surface_pressure']=aorta_tet10.element_set['Element_surface_pressure']
Boundary=aorta_tet10.node_set['boundary0'].tolist()+aorta_tet10.node_set['boundary1'].tolist()
aorta_tet10.mesh_data['Boundary']=torch.tensor(Boundary, dtype=torch.int64)
aorta_tet10.save_as_torch(filename+".pt")
save_polyhedron_mesh_to_inp(aorta_tet10, filename+".inp")#FEbio can import inp
