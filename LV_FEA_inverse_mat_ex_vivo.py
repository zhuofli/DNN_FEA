# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:20:53 2024

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
from PolyhedronMeshProcessing import PolyhedronMesh
import time
#%%
#attention:
#    run LV_FEA_QN_forward_inflation.py to generate mesh_px
#    mat_model for inverse_mat should be the same as the mat_model for inflation
#    this file only works for homogeneous/uniform mat distribution, i.e., all elements have the same mat 

all_mat=torch.load('D:/MLFEA/minliang_lv/data/125mat.pt')['mat_str']
matMean=torch.load('D:/MLFEA/minliang_lv/data/125mat.pt')['mean_mat_str']
px_pressure=20
mat_model='GOH_Jv'
#mat_true="1e2, 0, 1, 0, 0, 1e5"; mat_name='1e2'
mat_true=matMean; mat_name='matMean'
mesh_p0_str='D:/MLFEA/minliang_lv/data/ori/p54_c3d4_ori'
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
parser.add_argument('--lr', default=1, type=float)
parser.add_argument('--max_iter', default=500, type=int)
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
Mesh_X=PolyhedronMesh()
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
Mesh_x=PolyhedronMesh()
Mesh_x.load_from_torch(arg.mesh_px+".pt")
Node_x=Mesh_x.node.to(dtype).to(device)
print("loss1", Mesh_x.mesh_data['loss1'][-1])
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
    raise ValueError("this file does not supports :"+arg.mat_model)
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
#%%
LV_model=LVFEModel(Node_x, Element, Node_X, Boundary, Element_surface_pressure,
                   Mat_init, ElementOrientation, cal_1pk_stress, dtype, device, mode='inverse_mat')
pressure=arg.pressure
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
loss_list=[]
time_list=[]
Mat_list=[Mat_init.detach().cpu().numpy().tolist()]
t0=time.time()
#%%
optimizer = LBFGS([m_variable], lr=arg.lr, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20)
optimizer.set_strong_wolfe(t_max=1, verbose=False)
#optimizer.set_backtracking(t_list=[1, 0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-10], c=0.0001, verbose=False)
#%%
for iter1 in range(0, arg.max_iter):
    def closure(loss_fn="MSE"):
        Mat=get_Mat(m_variable)
        LV_model.set_material(Mat)
        out=LV_model.cal_energy_and_force(pressure)
        force_int=out['force_int']
        force_ext=out['force_ext']
        flag_valid=~(torch.isnan(force_int)|torch.isnan(force_ext)|torch.isinf(force_int)|torch.isinf(force_ext))
        loss=loss_function(force_int[flag_valid], force_ext[flag_valid], loss_fn)
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        return loss
    opt_cond=optimizer.step(closure)

    loss=float(closure(loss_fn="RMSE"))
    loss_list.append(loss)
    t1=time.time()
    time_list.append(t1-t0)
    
    if np.isnan(loss) == True or np.isinf(loss) == True:
        print(iter1, loss, t1-t0)
        print("abort: loss is nan or inf")
        sys.exit()

    Mat=get_Mat(m_variable)
    Mat_list.append(Mat.detach().cpu().numpy().reshape(-1).tolist())
    if len(Mat_list) > 2:
        Mat0=np.array(Mat_list[-2])
        Mat1=np.array(Mat_list[-1])
        a=np.abs(Mat0-Mat1).max()
        if a < 1e-6:
            print("a < 1e-6: opt_cond is True")
            opt_cond=True

    if iter1%100 == 0 or opt_cond==True:
        print(iter1, loss, t1-t0)
        print("Mat:", Mat_list[-1])
        display.clear_output(wait=False)
        fig, ax = plt.subplots()
        ax.plot(np.array(loss_list)/max(loss_list), 'r')
        ax.set_ylim(0, 1)
        ax.grid(True)
        display.display(fig)
        plt.close(fig)

    if opt_cond == True:
        print(iter1, loss, t1-t0)
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
filename=arg.folder_result+arg.mesh_px.split('/')[-1]+"_ex_vivo_mat"
torch.save({"arg":arg,
            "Mat":Mat_list,
            'Mat_ture':Mat_true,
            'Mat_pred':Mat_pred,
            "time":time_list},
           filename+".pt")
print("saved", filename)
#'''