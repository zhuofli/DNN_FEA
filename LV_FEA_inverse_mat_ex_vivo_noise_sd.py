# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 17:05:35 2024

@author: liang
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
sys.path.append("D:/MLFEA/code/pytorch_fea")
sys.path.append("D:/MLFEA/code/mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch_fea.optimizer.FE_lbfgs_ori import LBFGS
from PolyhedronMeshProcessing import PolyhedronMesh, TetrahedronMesh, Tet10Mesh
import time
#%%
#attention:
#    run LV_FEA_QN_forward_inflation.py to generate mesh_px
#    mat_model for inverse_mat should be the same as the mat_model for inflation

all_mat=torch.load('D:/MLFEA/minliang_lv/data/125mat.pt')['mat']
matMean=torch.load('D:/MLFEA/minliang_lv/data/125mat.pt')['mean_mat_str']
px_pressure=20
mat_model='GOH_Jv'
mat_true="1e2, 0, 1, 0, 0, 1e5"; mat_name='1e2'
#mat_true=matMean; mat_name='matMean'
mesh_p0_str='D:/MLFEA/minliang_lv/data/new_0220/p54_phase1_c3d4test'
#mesh_p0_str='D:/MLFEA/minliang_lv/data/new_0220/P54_phase1_lv_c3d10test'
mesh_px_str=mesh_p0_str+'_inflate_'+mat_model+'('+str(mat_name)+')_p'+str(px_pressure)
folder_result='D:/MLFEA/minliang_lv/data/new_0220/inverse_mat_ex_vivo'
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--folder_result', default=folder_result, type=str)
parser.add_argument('--mat_model', default=mat_model, type=str)
parser.add_argument('--mat_true', default=mat_true, type=str)#
parser.add_argument('--mesh_p0', default=mesh_p0_str, type=str)
parser.add_argument('--mesh_px', default=mesh_px_str, type=str)
parser.add_argument('--pressure', default=px_pressure, type=float)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--max_iter1', default=10000, type=int)
parser.add_argument('--noise_model', default='normal', type=str)#normal or uniform
parser.add_argument('--noise_level', default=0.02, type=float)
arg = parser.parse_args()
print(arg)
#%%
if arg.cuda >=0:
    device=torch.device("cuda:"+str(arg.cuda))
else:
    device=torch.device("cpu")
if arg.dtype == "float64":
    dtype=torch.float64
elif arg.dtype == "float32":
    dtype=torch.float32
else:
    raise ValueError("unkown dtype:"+arg.dtype)
#%%
if os.path.exists(arg.folder_result) == False:
    os.makedirs(arg.folder_result)
#%%
if 'c3d4' in arg.mesh_p0.lower():
    MeshType=TetrahedronMesh
elif 'c3d10' in arg.mesh_p0.lower(): 
    MeshType=Tet10Mesh
#%%
Mesh_X=MeshType()
Mesh_X.load_from_torch(arg.mesh_p0+".pt")
Node_X=Mesh_X.node.to(dtype).to(device)
Element=Mesh_X.element.to(device)
#%%
Boundary=Mesh_X.mesh_data['Boundary']
Boundary=torch.tensor(Boundary, dtype=torch.int64)
Element_surface_pressure=Mesh_X.mesh_data['Element_surface_pressure']
Element_surface_pressure=torch.tensor(Element_surface_pressure, dtype=torch.int64)
try:
    ElementOrientation=Mesh_X.element_data['orientation'].view(-1,3,3)
except:
    ElementOrientation=torch.eye(3).expand(len(Element),3,3)
#%%
mask=torch.ones_like(Node_X)
mask[Boundary]=0
#%%
Mesh_x=MeshType()
Mesh_x.load_from_torch(arg.mesh_px+".pt")
Node_x_clean=Mesh_x.node.to(dtype).to(device)
print("forward FEA loss1", Mesh_x.mesh_data['loss1'][-1])
#%% test the method using true stress
S_px_sd=Mesh_x.element_data['S'].to(dtype).to(device)
#%% add noise to Node_x_clean
Mesh_x.update_edge_length()
mean_edge_length=Mesh_x.edge_length.mean()
print('mean_edge_length', float(mean_edge_length))
if arg.noise_model == 'uniform':
    Node_x=Node_x_clean+arg.noise_level*mean_edge_length*torch.rand_like(Node_x_clean)
elif arg.noise_model == 'normal':
    Node_x=Node_x_clean+arg.noise_level*mean_edge_length*torch.randn_like(Node_x_clean)    
else:
    raise ValueError
#----------------------------------------------------------    
Mesh_x_noisy=PolyhedronMesh(Node_x, Mesh_x.element)
Mesh_x_noisy.save_as_vtk(arg.mesh_px+"_noise_"+arg.noise_model+str(arg.noise_level)+".vtk")
Mesh_x_noisy.save_as_torch(arg.mesh_px+"_noise_"+arg.noise_model+str(arg.noise_level)+".pt")
#%%
from LVFEModel import LVFEModel
#------------------
if arg.mat_model == "GOH":
    from torch_fea.material.Mat_GOH import cal_1pk_stress
elif arg.mat_model == "GOH_Fbar":
    from torch_fea.material.Mat_GOH_Fbar import cal_1pk_stress
elif arg.mat_model == "GOH_Jv":
    from torch_fea.material.Mat_GOH_Jv import cal_1pk_stress
elif arg.mat_model == "GOH_3Field":
    from torch_fea.material.Mat_GOH_3Field import cal_1pk_stress
else:
    raise ValueError("this file does not support :"+arg.mat_model)
#%%
m0_min=1;    m0_max=1000
m1_min=0;    m1_max=6000
m2_min=0.1;  m2_max=60
def get_Mat(m_variable):
    m0=m_variable[:,0]
    m1=m_variable[:,1]
    m2=m_variable[:,2]
    m3=m_variable[:,3]
    m4=m_variable[:,4]
    Mat=torch.zeros((1,6),dtype=dtype, device=device)
    Mat[0,0]=m0_min+(m0_max-m0_min)*torch.sigmoid(m0)
    Mat[0,1]=m1_min+(m1_max-m1_min)*torch.sigmoid(m1)
    Mat[0,2]=m2_min+(m2_max-m2_min)*torch.sigmoid(m2)
    Mat[0,3]=(1/3)*torch.sigmoid(m3)
    Mat[0,4]=(np.pi/2)*torch.sigmoid(m4)
    Mat[0,5]=1e5
    return Mat
#%%
m_variable=torch.zeros((1, 5), dtype=dtype, device=device, requires_grad=True)
Mat_init=get_Mat(m_variable)
print("Mat_init", Mat_init[0,:].detach().cpu().numpy().tolist())
#%%
def loss_function(A, B, reduction):
    Res=A-B
    if reduction == "MSE":
        loss=(Res**2).mean()
    elif reduction == "RMSE":
        loss=(Res**2).mean().sqrt()
    elif reduction == "MAE":
        loss=Res.abs().mean()
    elif reduction == "SSE":
        loss=(Res**2).sum()
    elif reduction == "SAE":
        loss=Res.abs().sum()
    return loss
#%%
flag_use_m_variable_only=False
flag_normalize_u_field=True
flag_use_loss2=True
#%%
u_field_init=Node_x-Node_X
alpha_u=u_field_init.abs().max().item()
print("alpha_u", alpha_u)
#%% u_variable is u_field or normalized u_field
u_variable=torch.zeros((Node_X.shape[0],3), dtype=dtype, device=device, requires_grad=True)
def get_u_field(u_variable):
    if flag_normalize_u_field == True:
        #assume noise is less than 0.5*alpha_u
        u_field=1.5*alpha_u*torch.tanh(u_variable)        
    else:
        u_field=u_variable
    return u_field
#%% initilize u_variable by u_field_init
optimizer = LBFGS([u_variable], lr=0.1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-5, tolerance_change=1e-10, history_size=20, max_iter=1)
for iter1 in range(0, 10000):
    def closure():
        u_field=get_u_field(u_variable)
        loss=loss_function(u_field, u_field_init, "MSE")
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        return loss
    opt_cond=optimizer.step(closure)
    if (iter1+1)%100 ==0 or opt_cond == True:
        u_field=get_u_field(u_variable)
        loss=loss_function(u_field, u_field_init, "RMSE")
        loss=float(loss)
        print("init u_field, loss", loss)
        if loss < 1e-12 or opt_cond==True:
            break
#%%
u_field=get_u_field(u_variable)
Node_x_pred=Node_X+u_field
noise_estimaiton=((Node_x_pred-Node_x)**2).mean().sqrt().item()
print('init noise_estimaiton', noise_estimaiton)        
#%%
Mat_list=[Mat_init.detach().cpu().numpy().tolist()]
time_list=[]
#%%
LV_model=LVFEModel(Node_x, Element, Node_X, Boundary, Element_surface_pressure,
                   Mat_init, ElementOrientation, cal_1pk_stress, dtype, device, mode='inverse_mat')
pressure=arg.pressure
#%% find the best mat among the 125 mat
'''
loss_list=[]
for m in range(0, len(all_mat)):
    mat_m=torch.tensor(all_mat[m].reshape(1,6), dtype=dtype, device=device)
    mat_m[:,4]=mat_m[:,4]*(np.pi/180)
    with torch.no_grad():
        LV_model.set_material(mat_m)        
        Spred=LV_model.cal_stress(stress='cauchy', create_graph=True)
        Spred=Spred.reshape(-1,9)           
        flag_valid=~(torch.isnan(Spred)|torch.isinf(Spred))
        flag_valid=Spred.abs() < 1e10        
        loss=loss_function(Spred[flag_valid], S_px_sd[flag_valid], "MSE")
    loss_list.append(float(loss))
best_idx=np.argmin(loss_list) 
print('Mat_init', all_mat[best_idx].tolist())
Mat_init=torch.tensor(all_mat[best_idx].reshape(1,6), dtype=dtype, device=device)
Mat_init[:,4]=Mat_init_best[:,4]*(np.pi/180)
if np.isnan(loss_list[best_idx]) or np.isinf(loss_list[best_idx]):
    print("Mat_init: loss is nan or inf")
    sys.exit()
'''    
#%% find the best m0 
#'''
loss_list=[]
m0_list=[]
for m0 in np.linspace(m0_min, m0_max, 1000):
    mat_m=torch.tensor([[m0, 1e-5, 1, 1/6, np.pi/4, 1e5]], dtype=dtype, device=device)    
    with torch.no_grad():
        LV_model.set_material(mat_m)        
        Spred=LV_model.cal_stress(stress='cauchy', create_graph=True)
        Spred=Spred.reshape(-1,9)           
        flag_valid=~(torch.isnan(Spred)|torch.isinf(Spred))
        flag_valid=Spred.abs() < 1e10        
        loss=loss_function(Spred[flag_valid], S_px_sd[flag_valid], "MSE")
    loss_list.append(float(loss))
    m0_list.append(m0)
best_idx=np.argmin(loss_list) 
Mat_init=torch.tensor([[m0_list[best_idx], 1e-5, 1, 1/6, np.pi/4, 1e5]], dtype=dtype, device=device)  
print('Mat_init', Mat_init.tolist())
if np.isnan(loss_list[best_idx]) or np.isinf(loss_list[best_idx]):
    print("Mat_init_best: loss is nan or inf")
    sys.exit()
#'''    
#%% re-initilize m_variable by Mat_init
optimizer = LBFGS([m_variable], lr=1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-5, tolerance_change=1e-10, history_size=20, max_iter=1)
for iter1 in range(0, 1000):
    def closure():
        Mat=get_Mat(m_variable)
        loss=loss_function(Mat, Mat_init, "MSE")
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        return loss
    opt_cond=optimizer.step(closure)
    if (iter1+1)%100 ==0 or opt_cond == True:
        Mat=get_Mat(m_variable)
        loss=loss_function(Mat, Mat_init, "MSE")
        loss=float(loss)
        print("Mat_init, loss", loss)
        if loss < 1e-12 or opt_cond==True:
            break   
#%%
optimizer = LBFGS([m_variable], lr=arg.lr, line_search_fn="strong_wolfe",
                 tolerance_grad=1e-5, tolerance_change=1e-10, history_size=20, max_iter=1)
optimizer.set_strong_wolfe(t_max=1, verbose=False)
#%%
print('start estimation')
t0=time.time()
for iter1 in range(0, arg.max_iter1):
    def closure(loss_fn="MSE", return_all=False):
        Mat=get_Mat(m_variable)
        LV_model.set_material(Mat)        
        Spred=LV_model.cal_stress(stress='cauchy', create_graph=True)
        Spred=Spred.reshape(-1,9)           
        flag_valid=~(torch.isnan(Spred)|torch.isinf(Spred))
        flag_valid=Spred.abs() < 1e10        
        loss=loss_function(Spred[flag_valid], S_px_sd[flag_valid], loss_fn)
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        if return_all == False:
            return loss
        else:
            return float(loss)
    opt_cond=optimizer.step(closure)

    t1=time.time()
    time_list.append(t1-t0)

    loss=closure(loss_fn="MSE", return_all=True)
    if np.isnan(loss) == True or np.isinf(loss) == True:
        print('abort: loss is nan or inf')
        sys.exit()
        
    Mat=get_Mat(m_variable)
    Mat_list.append(Mat.detach().cpu().numpy().reshape(-1).tolist())
    
    if len(Mat_list) > 100:
        Mat0=np.array(Mat_list[-100])
        Mat1=np.array(Mat_list[-1])
        a=np.abs(Mat0-Mat1).max()
        if a < 1e-6 and iter1 > 1000:
            opt_cond=True
            print('break: a < 1e-6 and iter1 > 1000')
            break

    if iter1%100 ==0 or iter1 == arg.max_iter1-1 or opt_cond==True:
        print("iter1", iter1, "loss", loss, "time", time_list[-1])
        print("Mat", Mat_list[-1])        
    
    if opt_cond == True:
        print(iter1, loss, t1-t0)
        print("break: opt_cond is True")
        break
