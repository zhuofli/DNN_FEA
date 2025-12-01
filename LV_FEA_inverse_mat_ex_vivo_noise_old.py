# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:03:37 2024

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
from torch.linalg import det, eigvalsh, matmul
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
mesh_px_str=mesh_p0_str+'_inflate_'+mat_model+'('+str(mat_name)+')_p'+str(px_pressure)+"std0"
folder_result='D:/MLFEA/minliang_lv/data/new_0220/inverse_mat_ex_vivo/'
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--folder_result', default=folder_result, type=str)
parser.add_argument('--mat_model', default=mat_model, type=str)
parser.add_argument('--mat_true', default=mat_true, type=str)#
parser.add_argument('--mesh_p0', default=mesh_p0_str, type=str)
parser.add_argument('--mesh_px', default=mesh_px_str, type=str)
parser.add_argument('--pressure', default=px_pressure, type=float)
parser.add_argument('--max_iter1', default=0, type=int)#mat only
parser.add_argument('--max_iter2', default=10000, type=int)#mat and u_field
parser.add_argument('--noise_model', default='normal', type=str)#normal or uniform
parser.add_argument('--noise_level', default=0.05, type=float) # 0~1, noise-to-signal(strain) ratio
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
#%%
from LVFEModel import LVFEModel
LV_model=LVFEModel(Node_x_clean, Element, Node_X, Boundary, Element_surface_pressure,
                   None, ElementOrientation, cal_1pk_stress, dtype, device, mode='inverse_mat')
pressure=arg.pressure
#%% add noise to Node_x_clean, and fix boundary
print("add noise to Node_x_clean")
#Mesh_x.update_edge_length()
#mean_edge_length=Mesh_x.edge_length.mean()
F=LV_model.cal_F_tensor()#(M,K,3,3)
C=matmul(F.permute(0,1,3,2), F)
E=0.5*(C-torch.eye(3).expand(1,1,3,3).to(dtype).to(device))
Ep = eigvalsh(E)
Ep=Ep.abs().max(dim=-1, keepdim=True)[0]
MeanMaxEp=Ep.mean().item()
print('MeanMaxEp', MeanMaxEp)
Mesh_x.update_edge_length()
mean_edge_length=Mesh_x.edge_length.mean().item()
print('mean_edge_length', mean_edge_length)
nosie_magnitude=arg.noise_level*MeanMaxEp*mean_edge_length
print('nosie_magnitude', nosie_magnitude)
print
if arg.noise_model == 'uniform':
    rng=np.random.RandomState(0)
    noise_x=rng.rand(len(Node_x_clean),3)    
    Node_x=Node_x_clean+nosie_magnitude*torch.tensor(noise_x, dtype=dtype, device=device)
    Node_x[Boundary]=Node_X[Boundary]
    noise_var=nosie_magnitude**2/12
    del noise_x, rng
elif arg.noise_model == 'normal':
    rng=np.random.RandomState(0)
    noise_x=rng.randn(len(Node_x_clean),3)
    Node_x=Node_x_clean+(nosie_magnitude/2)*torch.tensor(noise_x, dtype=dtype, device=device)    
    Node_x[Boundary]=Node_X[Boundary]
    noise_var=(nosie_magnitude/2)**2
    del noise_x, rng
else:
    raise ValueError
#Mesh_x_noisy=PolyhedronMesh(Node_x, Mesh_x.element)
#Mesh_x_noisy.save_as_vtk(arg.mesh_px+"_noise_"+arg.noise_model+str(arg.noise_level)+".vtk")
#Mesh_x_noisy.save_as_torch(arg.mesh_px+"_noise_"+arg.noise_model+str(arg.noise_level)+".pt")
#%% debug
LV_model.set_node_x(Node_x_clean)
F=LV_model.cal_F_tensor()
loss_F_clean=((det(F)-1)**2).mean().item()
print('loss_F_clean', loss_F_clean)
Mat_true=[float(m) for m in arg.mat_true.split(",")]
Mat_true[4]=(np.pi/180)*Mat_true[4]
print('Mat_true', Mat_true)
LV_model.set_node_x(Node_x_clean)
LV_model.set_material(torch.tensor([Mat_true]))
out=LV_model.cal_energy_and_force(pressure)
force_int=out['force_int']
force_ext=out['force_ext']
loss1_true=((force_int-force_ext)**2).mean().item()
loss2_true=((Node_x_clean-Node_x)**2).mean().item()
print("loss1_true", loss1_true, "loss2_true", loss2_true)
del loss1_true, loss2_true, F, loss_F_clean, Mat_true, out, force_int, force_ext
LV_model.set_node_x(Node_x)
LV_model.set_material(None)
#%% estimate u_field_init
print("estimate u_field_init : subject to J=1")
u_field_init=(Node_x-Node_X).detach()
u_field_init.requires_grad=True
optimizer = LBFGS([u_field_init], lr=1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
for iter1 in range(0, 1000):
    def closure(return_all=False):
        Node_x_pred=Node_X+u_field_init*mask
        LV_model.set_node_x(Node_x_pred)
        F=LV_model.cal_F_tensor()
        loss1=((det(F)-1)**2).mean()
        loss2=((Node_x_pred-Node_x)**2).mean()
        loss=loss1*1e3+loss2
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        if return_all == False:
            return loss
        else:
            return float(loss1), float(loss2)
    loss1, loss2=closure(True)
    loss1_threshold=1e-5
    if loss1 < 1e-5:
        print("u_field_init", iter1, "loss1", loss1, "<", loss1_threshold, ", set opt_cond to true", "loss2", loss2)
        opt_cond=True
        break
    opt_cond=optimizer.step(closure)    
    if (iter1+1)%100 ==0 or opt_cond == True:        
        loss1, loss2=closure(True)
        print("u_field_init", iter1, "loss1", loss1, "loss2", loss2)
    if opt_cond==True:
        break
del optimizer, loss1, loss2, opt_cond, iter1
#--------------------------------------        
u_field_init=u_field_init.detach()
#%%
flag_normalize_u_field=False
#%% u_variable is u_field or normalized u_field
u_variable=torch.zeros((Node_X.shape[0],3), dtype=dtype, device=device, requires_grad=True)
alpha_u=u_field_init.abs().max().item()
print("alpha_u", alpha_u)
def get_u_field(u_variable):
    if flag_normalize_u_field == True:
        #assume noise is less than 0.5*alpha_u
        u_field=1.5*alpha_u*torch.tanh(u_variable)        
    else:
        u_field=u_variable
    return u_field
#%% initialize u_variable by u_field_init
print("initialize u_variable by u_field_init")
optimizer = LBFGS([u_variable], lr=1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-20, tolerance_change=1e-20, history_size=20, max_iter=1)
for iter1 in range(0, 10000):
    def closure():
        u_field=get_u_field(u_variable)
        loss=((u_field-u_field_init)**2).mean()
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        return loss
    opt_cond=optimizer.step(closure)
    if iter1%100 ==0 or opt_cond == True:
        u_field=get_u_field(u_variable)        
        loss=((u_field-u_field_init)**2).mean().item()
        print("initialize u_variable, loss", loss)
        if loss < 1e-12 or opt_cond==True:
            break
del optimizer
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
Mat_init=Mat_init.detach()
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
#%% find the best mat among the 125 mat
'''
loss_list=[]
for m in range(0, len(all_mat)):
    mat_m=torch.tensor(all_mat[m].reshape(1,6), dtype=dtype, device=device)
    mat_m[:,4]=mat_m[:,4]*(np.pi/180)
    with torch.no_grad():
        LV_model.set_material(mat_m)
        out=LV_model.cal_energy_and_force(pressure)
    force_int=out['force_int']
    force_ext=out['force_ext']
    loss=loss_function(force_int, force_ext, "MSE")
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
'''
loss_list=[]
m0_list=[]
for m0 in np.linspace(m0_min, m0_max, 1000):
    mat_m=torch.tensor([[m0, 1e-5, 1, 1/6, np.pi/4, 1e5]], dtype=dtype, device=device)    
    with torch.no_grad():
        LV_model.set_material(mat_m)
        out=LV_model.cal_energy_and_force(pressure)
    force_int=out['force_int']
    force_ext=out['force_ext']
    loss=loss_function(force_int, force_ext, "MSE")
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
#'''
optimizer = LBFGS([m_variable], lr=1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-5, tolerance_change=1e-10, history_size=20, max_iter=1)
for iter1 in range(0, 0):
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
#'''        
#%%        
Mat_init=get_Mat(m_variable)
Mat_init=Mat_init.detach()
print("Mat_init", Mat_init[0,:].detach().cpu().numpy().tolist())
Mat_list=[Mat_init.detach().cpu().numpy().tolist()]
time_list=[]
#%%
optimizer = LBFGS([m_variable], lr=1, line_search_fn="strong_wolfe",
                 tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
optimizer.set_strong_wolfe(t_max=1, verbose=False)
#%%
print('start estimation - mat only')
t0=time.time()
#------------------------------
u_field=get_u_field(u_variable)
Node_x_pred=Node_X+u_field*mask
#LV_model.set_u_field(u_field.detach())
LV_model.set_node_x(Node_x_pred.detach())
del u_field, Node_x_pred
#------------------------------
for iter1 in range(0, arg.max_iter1):
    def closure(loss_fn="MSE", return_all=False):
        m_variable.data.clip_(min=-7, max=7)#avoid vanishing gradient
        Mat=get_Mat(m_variable)
        LV_model.set_material(Mat)
        out=LV_model.cal_energy_and_force(pressure)
        force_int=out['force_int']
        force_ext=out['force_ext']
        flag_valid=~(torch.isnan(force_int)|torch.isnan(force_ext)|torch.isinf(force_int)|torch.isinf(force_ext))
        #flag_valid=(force_int.abs() < 1e10)&(force_ext.abs() < 1e10)
        loss=loss_function(force_int[flag_valid], force_ext[flag_valid], loss_fn)
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
    del Mat
    
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
del optimizer    
#%%
class MyOptimizer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.optimizer1=LBFGS([m_variable], lr=1, line_search_fn="strong_wolfe",
                          tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
        self.optimizer1.set_strong_wolfe(t_max=1, verbose=False)
        self.optimizer2=LBFGS([u_variable], lr=1, line_search_fn="strong_wolfe",
                          tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
        self.optimizer2.set_strong_wolfe(t_max=1, verbose=False)
    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
    def step(self, closure):
        opt_cond1=self.optimizer1.step(closure)
        opt_cond2=self.optimizer2.step(closure)
        opt_cond = opt_cond1 and opt_cond2
        return opt_cond
    def reset_state(self):
        self.reset()
#%%
loss2_list=[]
#%%
optimizer = LBFGS([m_variable, u_variable], lr=1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
optimizer.set_strong_wolfe(t_max=1, verbose=False)
#-------------------------------------------------
#optimizer=MyOptimizer()
#%%
print('start estimation: mat and u_field')
t0=time.time()
for iter2 in range(0, arg.max_iter2):
    def closure(loss1_fn="MSE", loss2_fn="MSE", return_all=False):
        m_variable.data.clip_(min=-10, max=10)#avoid vanishing gradient
        Mat=get_Mat(m_variable)
        LV_model.set_material(Mat)
        u_field=get_u_field(u_variable)
        Node_x_pred=Node_X+u_field*mask
        #LV_model.set_u_field(u_field)
        LV_model.set_node_x(Node_x_pred.detach())
        out=LV_model.cal_energy_and_force(pressure)
        force_int=out['force_int']
        force_ext=out['force_ext']
        flag_valid=~(torch.isnan(force_int)|torch.isnan(force_ext)|torch.isinf(force_int)|torch.isinf(force_ext))
        #flag_valid=(force_int.abs() < 1e10)&(force_ext.abs() < 1e10)
        loss1=loss_function(force_int[flag_valid], force_ext[flag_valid], loss1_fn)
        loss2=loss_function(Node_x_pred, Node_x, loss2_fn)        
        #loss=loss1 + loss2*(1-arg.noise_level)*1e5 #not easy to adjust the weight of loss2
        #loss=loss1 + loss2/(1e-8+arg.noise_level**6)
        #loss=loss1 + 1e8*(loss2-noise_var).abs() #if we know noise level
        #loss=loss1 + 1e8*max(loss2, noise_var) #if we know noise level
        loss=loss1 + loss2
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
            
        if return_all == False:
            return loss
        else:
            return float(loss), float(loss1), float(loss2)
    opt_cond=optimizer.step(closure)

    t1=time.time()
    time_list.append(t1-t0)

    loss, loss1, loss2=closure(loss1_fn="MSE", loss2_fn="MSE", return_all=True)
    loss2_list.append(loss2)
    if np.isnan(loss) == True or np.isinf(loss) == True:
        print("iter2", iter2, "loss", loss, "loss1", loss1, "loss2", loss2)
        print("abort: loss is nan or inf")
        sys.exit()
    
    Mat=get_Mat(m_variable)
    Mat_list.append(Mat.detach().cpu().numpy().reshape(-1).tolist())
    
    if len(Mat_list) > 100 and len(loss2_list) > 2:
        Mat0=np.array(Mat_list[-2])
        Mat1=np.array(Mat_list[-1])
        a=np.abs(Mat0-Mat1).max()
        b=np.abs(loss2_list[-1]-loss2_list[-2])
        if a == 0 and b ==0:
            print('iter2', iter2, 'reset_state: a==0 and b==0')
            #optimizer.reset_state()
        #if a < 1e-12 and iter2 > 1000:
        #    opt_cond=True
        #    print('break: a < 1e-12 and iter2 > 1000')
        #    break

    if iter2%100 ==0 or iter2 == arg.max_iter2-1 or opt_cond==True:
        print("iter2", iter2, "loss", loss, "loss1", loss1, "loss2", loss2, "time", time_list[-1])
        print("Mat", Mat_list[-1])
        print('noise_var', noise_var)

    if opt_cond == True:
        print(iter2, loss, t1-t0)
        print("break: opt_cond is True")
        break
#%%
Mat_true=[float(m) for m in arg.mat_true.split(",")]
Mat_true[4]=(np.pi/180)*Mat_true[4]
Mat_pred=np.array(Mat_list[-1]).reshape(-1).tolist()
print("Mat_true",  Mat_true)
print("Mat_pred",  Mat_pred)
#%%
#'''
filename=arg.folder_result+arg.mesh_px.split('/')[-1]+'_ex_vivo_mat_noise_'+arg.noise_model+'('+str(arg.noise_level)+')'
torch.save({"arg":arg,
            "Mat":Mat_list,
            'Mat_ture':Mat_true,
            'Mat_pred':Mat_pred,
            "time":time_list},
           filename+".pt")
print("saved", filename)
#'''