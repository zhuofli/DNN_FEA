# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:38:03 2024

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
from PolyhedronMeshProcessing import TetrahedronMesh, Tet10Mesh
import time
#%%
#attention:
#    run LV_FEA_QN_forward_inflation.py to generate mesh_px
#    mat_model for inverse_mat should be the same as the mat_model for inflation
#    this file only works for homogeneous/uniform mat distribution, i.e., all elements have the same mat 

all_mat=torch.load('D:/MLFEA/minliang_lv/data/125mat.pt')['mat']
matMean=torch.load('D:/MLFEA/minliang_lv/data/125mat.pt')['mean_mat_str']
px_pressure=20
mat_model='GOH_Jv'
#mat_true="1e2, 0, 1, 0, 0, 1e5"; mat_name='1e2'
mat_true=matMean; mat_name='matMean'
mesh_p0_str='D:/MLFEA/minliang_lv/data/ori/x4_c3d4_ori'
mesh_px_str=mesh_p0_str+'_inflate_'+mat_model+'('+str(mat_name)+')_p'+str(px_pressure)
folder_result='D:/MLFEA/minliang_lv/data/ori/inverse_mat_ex_vivo/'
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
parser.add_argument('--max_iter2', default=100000, type=int)#mat and u_field
parser.add_argument('--noise_model', default='normal', type=str)#normal or uniform
parser.add_argument('--noise_level', default=0, type=float) # 0~1, noise-to-signal(strain) ratio
parser.add_argument('--random_seed', default=1, type=int)
parser.add_argument('--lossJ_threshold', default=3e-5, type=float)
parser.add_argument('--beta_range', default=[1e-8, 1e8], type=list)#[beta_min(1e-8), beta_max(1e8)]
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
    TetMesh=TetrahedronMesh
elif 'c3d10' in arg.mesh_p0.lower(): 
    TetMesh=Tet10Mesh
#%%
Mesh_X=TetMesh()
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
    print("ElementOrientation is not available, set it to identity matrix")
#%%
mask=torch.ones_like(Node_X)
mask[Boundary]=0
#%%
pressure_node_idx_list=[]
for n in range(len(Element_surface_pressure)):
    pressure_node_idx_list.extend(Element_surface_pressure[n])    
pressure_node_idx_list=np.unique(pressure_node_idx_list).tolist()    
non_pressure_node_idx_list=list(set(np.arange(0, Node_X.shape[0]))-set(pressure_node_idx_list))
#%%
Mesh_x=TetMesh()
Mesh_x.load_from_torch(arg.mesh_px+".pt")
Node_x_clean=Mesh_x.node.to(dtype).to(device)
print("forward FEA loss1", Mesh_x.mesh_data['loss1'][-1])
#%%
from LVFEModel import LVFEModel
LV_model=LVFEModel(Node_X, Element, Node_X, Boundary, Element_surface_pressure,
                   None, ElementOrientation, cal_1pk_stress, dtype, device, mode='inverse_mat')
pressure=arg.pressure
#%% add noise to Node_x_clean, and fix boundary
print("add noise to Node_x_clean")
LV_model.set_node_x(Node_x_clean)
F=LV_model.cal_F_tensor()#(M,K,3,3)
C=matmul(F.permute(0,1,3,2), F)
E=0.5*(C-torch.eye(3).expand(1,1,3,3).to(dtype).to(device))
Ep = eigvalsh(E)
Ep=Ep.abs().max(dim=-1, keepdim=True)[0]
MeanMaxEp=Ep.mean().item()
print('MeanMaxEp', MeanMaxEp)
Mesh_X.update_edge_length()
mean_edge_length=Mesh_X.edge_length.mean().item()
print('mean_edge_length', mean_edge_length)
nosie_magnitude=arg.noise_level*MeanMaxEp*mean_edge_length
print('nosie_magnitude', nosie_magnitude)
print
if arg.noise_model == 'uniform':
    rng=np.random.RandomState(arg.random_seed)
    noise_x=1-2*rng.rand(len(Node_x_clean),3)
    Node_x=Node_x_clean+nosie_magnitude*torch.tensor(noise_x, dtype=dtype, device=device)
    Node_x[Boundary]=Node_X[Boundary]
    noise_var=(2*nosie_magnitude)**2/12
    del noise_x, rng
elif arg.noise_model == 'normal':
    rng=np.random.RandomState(arg.random_seed)
    noise_x=rng.randn(len(Node_x_clean),3)
    Node_x=Node_x_clean+nosie_magnitude*torch.tensor(noise_x, dtype=dtype, device=device)    
    Node_x[Boundary]=Node_X[Boundary]
    noise_var=nosie_magnitude**2
    del noise_x, rng
else:
    raise ValueError
print('noise_var', noise_var)    
#Mesh_x_noisy=TetMesh(Node_x, Mesh_x.element)
#Mesh_x_noisy.save_as_vtk(arg.mesh_px+"_noise_"+arg.noise_model+str(arg.noise_level)+".vtk")
#Mesh_x_noisy.save_as_torch(arg.mesh_px+"_noise_"+arg.noise_model+str(arg.noise_level)+".pt")
LV_model.set_node_x(Node_x)
#%%
def cal_loss(A, B, reduction):
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
#%% initialize u_field
print("initialize u_field: subject to J=1")
#Mat_init=torch.tensor([[500, 0, 1, 1/6, np.pi/4, 1e5]], dtype=dtype, device=device)
Mat_init=get_Mat(torch.zeros((1, 5), dtype=dtype, device=device))
print('Mat_init', Mat_init.tolist())
u_field=(Node_x-Node_X).detach()
u_field.requires_grad=True
optimizer = LBFGS([u_field], lr=1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-20, tolerance_change=1e-20, history_size=20, max_iter=1)
lossJ_threshold=arg.lossJ_threshold
beta=1
beta_min, beta_max=arg.beta_range
LV_model.set_material(Mat_init)
for iter1 in range(0, 1000):
    def closure(return_all=False):
        Node_x_pred=Node_X+u_field*mask
        LV_model.set_node_x(Node_x_pred)      
        F=LV_model.cal_F_tensor()
        out=LV_model.cal_energy_and_force(pressure)        
        force_int=out['force_int']
        force_ext=out['force_ext']
        loss1=cal_loss(force_int, force_ext, "MSE")        
        loss2=cal_loss(Node_x_pred, Node_x, "MSE")        
        lossJ=cal_loss(det(F), 1, "MSE")        
        loss=lossJ+loss2*beta #do not use loss1 here
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        if return_all == False:
            return loss
        else:
            return float(loss1), float(loss2), float(lossJ)
    opt_cond=optimizer.step(closure)    

    loss1, loss2, lossJ=closure(True)        
    if np.isinf(loss1) or np.isnan(loss1) or loss2 < noise_var:
        beta=beta*0.9
    else:
        beta=beta*1.1
    beta=max(min(beta, beta_max), beta_min)
    
    if (not np.isinf(loss1)) and (not np.isnan(loss1)) and lossJ < lossJ_threshold:
        print("u_field init: lossJ < lossJ_threshold("+str(lossJ_threshold)+")")
        opt_cond=True

    if iter1%100 ==0 or opt_cond == True:        
        print("u_field init: iter1", iter1, "loss1", loss1, "loss2", loss2, "lossJ", lossJ, "beta", beta)
    if opt_cond==True:
        break
print("u_field init: iter1", iter1, "loss1", loss1, "loss2", loss2, "lossJ", lossJ, "beta", beta)    
if np.isinf(loss1) or np.isnan(loss1):
    print("abort: u_field init: loss1 is nan or inf")
    sys.exit()    
#%%    
del optimizer, loss1, loss2, lossJ, opt_cond, iter1
#%% find a good Mat_init that will not lead to loss=inf
'''
print('find a good Mat_init')
# find the best m0 
loss_m0_list=[]
m0_list=[]
for m0 in np.linspace(m0_min+0.01*(m0_max-m0_min), m0_max-0.01*(m0_max-m0_min), 100):
    mat_m=torch.tensor([[m0, 0, 1, 1/6, np.pi/4, 1e5]], dtype=dtype, device=device)    
    with torch.no_grad():
        LV_model.set_material(mat_m)
        out=LV_model.cal_energy_and_force(pressure)
    force_int=out['force_int']
    force_ext=out['force_ext']
    loss_a=cal_loss(force_int[pressure_node_idx_list], force_ext[pressure_node_idx_list], "MSE")
    loss_b=cal_loss(force_int[non_pressure_node_idx_list], force_ext[non_pressure_node_idx_list], "MSE")
    loss=(loss_a+loss_b)/2
    loss_m0_list.append(float(loss))
    m0_list.append(m0)
best_idx=np.argmin(loss_m0_list) 
Mat_init=torch.tensor([[m0_list[best_idx], 0, 1, 1/6, np.pi/4, 1e5]], dtype=dtype, device=device)  
print('Mat_init', Mat_init.tolist(), "loss", loss_m0_list[best_idx])
if np.isnan(loss_m0_list[best_idx]) or np.isinf(loss_m0_list[best_idx]):
    print("abort: Mat_init: loss is nan or inf")
    sys.exit()
'''
#%% 
m_variable=torch.zeros((1, 5), dtype=dtype, device=device, requires_grad=True)
'''
print("initialize m_variable by Mat_init")
optimizer = LBFGS([m_variable], lr=1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
for iter1 in range(0, 1000):
    def closure():
        Mat=get_Mat(m_variable)
        loss=cal_loss(Mat, Mat_init, "MSE")
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        return loss
    opt_cond=optimizer.step(closure)
    if (iter1+1)%100 ==0 or opt_cond == True:        
        loss=float(closure())
        print("m_variable: iter1", iter1, "loss", loss)
        if loss < 1e-12 or opt_cond==True:
            break
'''
#%%
print('start estimation - mat only')
Mat_init=get_Mat(m_variable)
print('Mat_init', Mat_init.tolist())
Mat_list=[Mat_init.detach().cpu().numpy().reshape(-1).tolist()]
time_list_mat_only=[]
loss_list_mat_only=[]
#%%
optimizer = LBFGS([m_variable], lr=1, line_search_fn="strong_wolfe",
                 tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
optimizer.set_strong_wolfe(t_max=1, verbose=False)
#%%
Node_x_pred=Node_X+u_field*mask
LV_model.set_node_x(Node_x_pred.detach())
del Node_x_pred
for iter1 in range(0, arg.max_iter1):
    def closure():
        m_variable.data.clip_(min=-10, max=10)#avoid vanishing gradient
        Mat=get_Mat(m_variable)
        LV_model.set_material(Mat)
        out=LV_model.cal_energy_and_force(pressure)
        force_int=out['force_int']
        force_ext=out['force_ext']
        loss=cal_loss(force_int, force_ext, "MSE")
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        return loss
    t0=time.time()
    opt_cond=optimizer.step(closure) 
    t1=time.time()
    time_list_mat_only.append(t1-t0)
    #
    loss=float(closure())
    loss_list_mat_only.append(loss)
    if np.isnan(loss) == True or np.isinf(loss) == True:
        print('abort: loss is nan or inf')
        sys.exit()
    #    
    Mat=get_Mat(m_variable)
    Mat_list.append(Mat.detach().cpu().numpy().reshape(-1).tolist())
    del Mat
    #
    if len(Mat_list) > 100:
        Mat0=np.array(Mat_list[-2])
        Mat1=np.array(Mat_list[-1])
        a=np.abs(Mat0-Mat1).max()
        if a == 0:
            opt_cond=True
            print('break: a < 1e-6 and iter1 > 1000')            

    if iter1%100 ==0 or iter1 == arg.max_iter1-1 or opt_cond==True:
        print("iter1", iter1, "loss", loss, "time", sum(time_list_mat_only))
        print("Mat", Mat_list[-1])
    
    if opt_cond == True:
        print(iter1, loss, t1-t0)
        print("break: opt_cond is True")
        break
#%%
del optimizer    
#%%
print('start estimation: mat and u_field')
time_list=[]
loss1_list=[]
loss2_list=[]
beta=1
#%%
class MyOptimizer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.optimizer1=LBFGS([m_variable], lr=1, line_search_fn="strong_wolfe",
                          tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
        self.optimizer1.set_strong_wolfe(t_max=1, verbose=False)
        self.optimizer2=LBFGS([u_field], lr=1, line_search_fn="strong_wolfe",
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
optimizer = LBFGS([m_variable, u_field], lr=1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
optimizer.set_strong_wolfe(t_max=1, verbose=False)
#-------------------------------------------------
#optimizer=MyOptimizer()#bad m_variable got stuck at -10
beta_min, beta_max=arg.beta_range
#%%
for iter2 in range(0, arg.max_iter2):    
    #
    def closure(return_all=False):
        m_variable.data.clip_(min=-10, max=10)#avoid vanishing gradient
        Mat=get_Mat(m_variable)
        LV_model.set_material(Mat)
        Node_x_pred=Node_X+u_field*mask
        LV_model.set_node_x(Node_x_pred)
        out=LV_model.cal_energy_and_force(pressure)
        force_int=out['force_int']
        force_ext=out['force_ext']
        loss1=cal_loss(force_int, force_ext, "MSE")
        loss2=cal_loss(Node_x_pred, Node_x, "MSE")        
        loss=loss1 + loss2*beta
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
            #u_field.grad.data-=u_field.grad.data-((force_int-force_ext)+(Node_x_pred-Node_X))
        if return_all == False:
            return loss
        else:
            return float(loss), float(loss1), float(loss2)
    t0=time.time()
    opt_cond=optimizer.step(closure)
    t1=time.time()
    time_list.append(t1-t0)
    
    Node_x_pred=Node_X+u_field*mask
    node_error=((Node_x_pred-Node_x_clean)**2).mean().item()
    del Node_x_pred
    
    loss, loss1, loss2=closure(return_all=True)
    loss1_list.append(loss1)
    loss2_list.append(loss2)
    if np.isnan(loss) == True or np.isinf(loss) == True:
        print("iter2", iter2, "loss", loss, "loss1", loss1, "loss2", loss2, "node_error", node_error)
        print("abort: loss is nan or inf")
        sys.exit()

    '''
    delta=(loss2+1e-8)/(noise_var+1e-8)-1
    if delta > 0:
        if abs(delta) > 0.05:        
            beta=beta+delta*1000
        else:
            beta=beta+delta*100
    else:
        beta=beta+delta*1000
    '''
    if len(loss2_list) >= 100:
        if loss2 > noise_var and loss2_list[-1] >= loss2_list[-10]:
            beta=beta*(loss2+1e-8)/(noise_var+1e-8)    
        elif loss2 < noise_var and loss2_list[-1] <= loss2_list[-10]:
            beta=beta*(loss2+1e-8)/(noise_var+1e-8)
    beta=max(min(beta, beta_max), beta_min)

    Mat=get_Mat(m_variable)
    Mat_list.append(Mat.detach().cpu().numpy().reshape(-1).tolist())
    
    if len(Mat_list) > 100 and len(loss2_list) > 2:
        Mat0=np.array(Mat_list[-2])
        Mat1=np.array(Mat_list[-1])
        a=np.abs(Mat0-Mat1).max()
        b=np.abs(loss2_list[-1]-loss2_list[-2])
        if a == 0 and b ==0 and iter2%20 == 0:
            print('iter2', iter2, 'reset_state: a==0 and b==0')
            print("iter2", iter2, "loss", loss, "loss1", loss1, "loss2", loss2, "time", sum(time_list))
            optimizer.reset_state()

    if iter2%100 ==0 or iter2 == arg.max_iter2-1 or opt_cond==True:
        print("iter2", iter2, "loss", loss, "loss1", loss1, "loss2", loss2, "time", sum(time_list))        
        print('noise_var', noise_var, "beta", beta, "node_error", node_error)
        print("Mat", Mat_list[-1])

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
Node_x_pred=(Node_X+u_field*mask).detach().cpu()
#%% save
filename=(arg.folder_result+arg.mesh_px.split('/')[-1]+'_ex_vivo_mat_noise_'
          +arg.noise_model+'('+str(arg.noise_level)+')'+"_s"+str(arg.random_seed))
torch.save({"arg":arg,
            "noise_var":noise_var,
            "Mat":Mat_list,
            'Mat_ture':Mat_true,
            'Mat_pred':Mat_pred,
            'Node_x_pred':Node_x_pred,
            'time_list_mat_only':time_list_mat_only,
            'loss_list_mat_only':loss_list_mat_only,
            "time":time_list,
            'loss1_list':loss1_list,
            'loss2_list':loss2_list},
           filename+".pt")
print("saved", filename)
