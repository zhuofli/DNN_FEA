# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:16:08 2024

@author: liang
"""
import torch
import numpy as np
import sys
sys.path.append("D:/MLFEA/code/mesh")
from PolyhedronMeshProcessing import PolyhedronMesh, ComputeAngleBetweenTwoVectorIn3D
#%%
def generate_mat_distribution(idx, file_mesh_p0):
    if idx==1:
        return generate_mat_distribution_1(file_mesh_p0)
    elif idx==2:
        return generate_mat_distribution_2(file_mesh_p0)
    elif idx==3:
        return generate_mat_distribution_3(file_mesh_p0)
    elif idx==4:
        return generate_mat_distribution_4(file_mesh_p0)
    elif idx==4.1:
        return generate_mat_distribution_4_1(file_mesh_p0)
    elif idx==4.2:
        return generate_mat_distribution_4_2(file_mesh_p0)
    elif idx==4.5:
        return generate_mat_distribution_4_noise(file_mesh_p0)
    elif idx==5:
        return generate_mat_distribution_5(file_mesh_p0)
    elif idx==6:
        return generate_mat_distribution_6(file_mesh_p0)
    else:
        raise ValueError

def generate_mat_distribution_1(file_mesh_p0):
    #GOH mat_model
    mesh=PolyhedronMesh()
    mesh.load_from_torch(file_mesh_p0+'.pt')
    Mat=torch.zeros((len(mesh.element), 6), dtype=torch.float64)
    rng=np.random.RandomState(0)
    Mat[:,0]=1+999*torch.tensor(rng.rand(len(mesh.element)), dtype=torch.float64)
    Mat[:,2]=1
    Mat[:,5]=1e5
    return Mat

def generate_mat_distribution_2(file_mesh_p0):
    #GOH mat_model
    mesh=PolyhedronMesh()
    mesh.load_from_torch(file_mesh_p0+'.pt')
    Mat=torch.zeros((len(mesh.element), 6), dtype=torch.float64)
    #computer center of each element
    center=mesh.node[mesh.element].mean(dim=1)
    #generate mat
    Mat[:,0]=200+100*(0.2*torch.sin(center[:,2])+0.3*torch.sin(0.1*center[:,2])+0.5*torch.sin(0.01*center[:,2]))
    Mat[:,2]=1
    Mat[:,5]=1e5
    return Mat

def generate_mat_distribution_3(file_mesh_p0):
    #GOH mat_model
    mesh=PolyhedronMesh()
    mesh.load_from_torch(file_mesh_p0+'.pt')
    Mat=torch.zeros((len(mesh.element), 6), dtype=torch.float64)
    #computer center of each element
    center=mesh.node[mesh.element].mean(dim=1)
    #generate mat
    A=0.2*torch.sin(center[:,2])+0.3*torch.sin(0.1*center[:,2])+0.5*torch.sin(0.01*center[:,2])
    temp1=center.clone(); temp1[:,2]=0
    temp2=torch.zeros_like(center); temp2[:,1]=1    
    theta=ComputeAngleBetweenTwoVectorIn3D(temp1, temp2)
    #B=0.2*torch.sin(10*theta)+0.3*torch.sin(theta)+0.5*torch.sin(0.1*theta)
    B=torch.sin(10*theta)
    Mat[:,0]=200+100*A*B
    Mat[:,2]=1
    Mat[:,5]=1e5
    return Mat

def generate_mat_distribution_4(file_mesh_p0):
    #HO mat_model
    mesh=PolyhedronMesh()
    mesh.load_from_torch(file_mesh_p0+'.pt')
    Mat=torch.zeros((len(mesh.element), 6), dtype=torch.float64)
    #computer center of each element
    center=mesh.node[mesh.element].mean(dim=1)
    #generate mat
    A=0.2*torch.sin(1*center[:,2])+0.3*torch.sin(0.1*center[:,2])+0.5*torch.sin(0.01*center[:,2])
    temp1=center.clone(); temp1[:,2]=0
    temp2=torch.zeros_like(center); temp2[:,1]=1    
    theta=ComputeAngleBetweenTwoVectorIn3D(temp1, temp2)
    #B=0.2*torch.sin(10*theta)+0.3*torch.sin(theta)+0.5*torch.sin(0.1*theta)
    B=torch.sin(10*theta)
    Mat[:,0]=0.18+0.2*A*B
    Mat[:,1]=2.6-0.5*A*B
    Mat[:,2]=3.34+0.5*A*B
    Mat[:,3]=2.73-0.5*A*B
    Mat[:,4]=1e5
    return Mat

def generate_mat_distribution_4_1(file_mesh_p0):
    #HO mat_model
    mesh=PolyhedronMesh()
    mesh.load_from_torch(file_mesh_p0+'.pt')
    Mat=torch.zeros((len(mesh.element), 6), dtype=torch.float64)
    #computer center of each element
    center=mesh.node[mesh.element].mean(dim=1)
    #generate mat
    #A=0.2*torch.sin(0.5*center[:,2])+0.3*torch.sin(0.1*center[:,2])+0.5*torch.sin(0.01*center[:,2])
    temp1=center.clone(); temp1[:,2]=0
    temp2=torch.zeros_like(center); temp2[:,1]=1    
    theta=ComputeAngleBetweenTwoVectorIn3D(temp1, temp2)
    #B=0.2*torch.sin(10*theta)+0.3*torch.sin(theta)+0.5*torch.sin(0.1*theta)
    #B=torch.sin(5*theta)
    Mat[:,0]=2.28#+0.5*A*B
    Mat[:,1]=9.726#-0.5*A*B
    Mat[:,2]=1.685#+0.5*A*B
    Mat[:,3]=15.779#-0.5*A*B
    Mat[:,4]=1e5
    return Mat

def generate_mat_distribution_4_2(file_mesh_p0):
    #HO mat_model
    mesh=PolyhedronMesh()
    mesh.load_from_torch(file_mesh_p0+'.pt')
    Mat=torch.zeros((len(mesh.element), 6), dtype=torch.float64)
    #computer center of each element
    center=mesh.node[mesh.element].mean(dim=1)
    #generate mat
    A=0.2*torch.sin(0.5*center[:,2])+0.3*torch.sin(0.1*center[:,2])+0.5*torch.sin(0.01*center[:,2])
    temp1=center.clone(); temp1[:,2]=0
    temp2=torch.zeros_like(center); temp2[:,1]=1    
    theta=ComputeAngleBetweenTwoVectorIn3D(temp1, temp2)
    #B=0.2*torch.sin(10*theta)+0.3*torch.sin(theta)+0.5*torch.sin(0.1*theta)
    B=torch.sin(5*theta)
    Mat[:,0]=0.45+0.5*A*B
    Mat[:,1]=7.21-0.5*A*B
    Mat[:,2]=15.19+0.5*A*B
    Mat[:,3]=20.42-0.5*A*B
    Mat[:,4]=1e5
    return Mat

def generate_mat_distribution_4_noise(file_mesh_p0):
    #HO mat_model
    mesh=PolyhedronMesh()
    mesh.load_from_torch(file_mesh_p0+'.pt')
    Mat=torch.zeros((len(mesh.element), 6), dtype=torch.float64)
    #computer center of each element
    center=mesh.node[mesh.element].mean(dim=1)
    #generate mat
    A=0.2*torch.sin(1*center[:,2])+0.3*torch.sin(0.1*center[:,2])+0.5*torch.sin(0.01*center[:,2])
    temp1=center.clone(); temp1[:,2]=0
    temp2=torch.zeros_like(center); temp2[:,1]=1    
    theta=ComputeAngleBetweenTwoVectorIn3D(temp1, temp2)
    #B=0.2*torch.sin(10*theta)+0.3*torch.sin(theta)+0.5*torch.sin(0.1*theta)
    B=torch.sin(10*theta)
    noiselevel=0.003
    noise0=np.random.normal(0,noiselevel, size=Mat[:,0].shape)
    noise0 = np.clip(noise0, -3*noiselevel, 3*noiselevel)
    #noise1=np.random.normal(0.05,noiselevel, size=Mat[:,0].shape)
    noise2=np.random.normal(0,noiselevel, size=Mat[:,0].shape)
    noise2 = np.clip(noise2, -3*noiselevel, 3*noiselevel)
    #noise3=np.random.normal(1,noiselevel, size=Mat[:,0].shape)
    Mat[:,0]=0.18*(1+noise0)+0.2*A*B
    Mat[:,1]=(2.6-0.5*A*B)#*(1+noise2)
    Mat[:,2]=3.34*(1+noise0)+0.5*A*B
    Mat[:,3]=2.73-0.5*A*B
    Mat[:,4]=1e5
    return Mat

def generate_mat_distribution_5(file_mesh_p0):
    #NH mat_model
    mesh=PolyhedronMesh()
    mesh.load_from_torch(file_mesh_p0+'.pt')
    Mat=torch.zeros((len(mesh.element), 6), dtype=torch.float64)
    #computer center of each element
    center=mesh.node[mesh.element].mean(dim=1)
    #generate mat
    A=0.2*torch.sin(1*center[:,2])+0.3*torch.sin(0.1*center[:,2])+0.5*torch.sin(0.01*center[:,2])
    temp1=center.clone(); temp1[:,2]=0
    temp2=torch.zeros_like(center); temp2[:,1]=1    
    theta=ComputeAngleBetweenTwoVectorIn3D(temp1, temp2)
    #B=0.2*torch.sin(10*theta)+0.3*torch.sin(theta)+0.5*torch.sin(0.1*theta)
    B=torch.sin(5*theta)
    Mat[:,0]=344+50*A*B
    Mat[:,1]=3000-100*A*B
    
    return Mat

def generate_mat_distribution_6(file_mesh_p0):
    #HO mat_model
    mesh=PolyhedronMesh()
    mesh.load_from_torch(file_mesh_p0+'.pt')
    Mat=torch.zeros((len(mesh.element), 9), dtype=torch.float64)
    #computer center of each element
    center=mesh.node[mesh.element].mean(dim=1)
    #generate mat
    A=0.2*torch.sin(0.5*center[:,2])+0.3*torch.sin(0.1*center[:,2])+0.5*torch.sin(0.01*center[:,2])
    temp1=center.clone(); temp1[:,2]=0
    temp2=torch.zeros_like(center); temp2[:,1]=1    
    theta=ComputeAngleBetweenTwoVectorIn3D(temp1, temp2)
    #B=0.2*torch.sin(10*theta)+0.3*torch.sin(theta)+0.5*torch.sin(0.1*theta)
    B=torch.sin(5*theta)
    Mat[:,0]=0.654+0.5*A*B
    Mat[:,1]=20.85-0.5*A*B
    Mat[:,2]=6.959+0.5*A*B
    Mat[:,3]=2.15-0.5*A*B
    Mat[:,4]=3.407 
    Mat[:,5]=1.05
    Mat[:,6]=1.219
    Mat[:,7]=21.79
    Mat[:,8]=1e5
    return Mat