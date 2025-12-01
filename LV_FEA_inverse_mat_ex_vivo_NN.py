# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:22:07 2024

@author: liang
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
sys.path.append("E:/Research/NNFEA/code")
sys.path.append("E:/Research/NNFEA/code/pytorch_fea")
sys.path.append("E:/Research/NNFEA/code/mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch_fea.utils.functions import cal_attribute_on_node, cal_von_mises_stress
from torch_fea.optimizer.FE_lbfgs_ori import LBFGS
from MatNet import Net0, Net3
from PolyhedronMeshProcessing import PolyhedronMesh, TetrahedronMesh, Tet10Mesh
import time
from LVFEModel import LVFEModel
from LV_mat_distribution import generate_mat_distribution
#%%
#attention:
#    run LV_FEA_QN_forward_inflation.py to generate mesh_px
#    mat_model for inverse_mat should be the same as the mat_model for inflation
#    this file works for heterogeneous/nonuniform mat distribution
#
all_mat=torch.load('E:/Research/NNFEA/data/125mat.pt')['mat_str']
matMean=torch.load('E:/Research/NNFEA/data/125mat.pt')['mean_mat_str']
px_pressure=20
mat_model='GOH_Jv'
#mat_true="1e2, 0, 1, 0, 0, 1e5"; mat_name='1e2'
#mat_true=matMean; mat_name='matMean'
mat_true='generate_mat_distribution(3,arg.mesh_p0)'; mat_name='distribution3'
mesh_p0_str='E:/Research/NNFEA/data/ori/p54_c3d4_ori'
#mesh_p0_str='D:/MLFEA/minliang_lv/data/new_0220/P54_phase1_lv_c3d10test'
mesh_px_str=mesh_p0_str+'_inflate_'+mat_model+'('+str(mat_name)+')_p'+str(px_pressure)
folder_result='E:/Research/NNFEA/data/ori/inverse_mat_ex_vivo_NN'
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--folder_result', default=folder_result, type=str)
parser.add_argument('--mesh_p0', default=mesh_p0_str, type=str)
parser.add_argument('--mesh_px', default=mesh_px_str, type=str)
parser.add_argument('--mesh_input', default=mesh_p0_str, type=str)#template
parser.add_argument('--mat_model', default=mat_model, type=str)
parser.add_argument('--mat_true', default=mat_true, type=str)
parser.add_argument('--pressure', default=px_pressure, type=float)
parser.add_argument('--max_iter1', default=1000, type=int)
parser.add_argument('--max_iter2', default=10000, type=int)
parser.add_argument('--net', default="Net3(3,256,2,1,1,0,5)", type=str)
#parser.add_argument('--net', default="Net0(3,256,4,1,1,0,5)", type=str)
#parser.add_argument('--net', default="none", type=str)
arg = parser.parse_args()
print(arg)
#%%
if arg.cuda >=0:
    device=torch.device("cpu")
    #device=torch.device("cuda:"+str(arg.cuda))
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
#%%
Mesh_x=TetMesh()
Mesh_x.load_from_torch(arg.mesh_px+".pt")
Node_x=Mesh_x.node.to(dtype).to(device)
#%%
if len(arg.mesh_input)>0:
    mesh_input=PolyhedronMesh()
    mesh_input.load_from_torch(arg.mesh_input+".pt")
else:
    mesh_input=Mesh_X
    print('Mesh_X is mesh_input')
if arg.mesh_input == arg.mesh_p0:
    print('Mesh_X is mesh_input')
NodeInput=mesh_input.node[mesh_input.element].mean(dim=1)
NodeInput=NodeInput.to(dtype).to(device)
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
    raise ValueError("this file does not supports :"+arg.mat_model)
#%%
Mat_true=eval(arg.mat_true)
Mat_true=Mat_true.detach()
Mat_true[:,4]=(np.pi/180)*Mat_true[:,4]
Mat_true=Mat_true.to(device)    
#%%
def process_raw_mat(Mat_raw):
    m0_min=1;    m0_max=1000
    m1_min=0;    m1_max=6000
    m2_min=0.1;  m2_max=60
    M=torch.sigmoid(Mat_raw)
    Mat=torch.zeros((Element.shape[0], 6), dtype=dtype, device=device)
    Mat[:,0]=m0_min+(m0_max-m0_min)*M[:,0]
    Mat[:,1]=0#m1_min+(m1_max-m1_min)*M[:,1]
    Mat[:,2]=1#m2_min+(m2_max-m2_min)*M[:,2]
    Mat[:,3]=0#(1/3)*M[:,3]
    Mat[:,4]=0#(np.pi/2)*M[:,4]
    Mat[:,5]=1e5 # a known constant
    return Mat
#%%
LV_model=LVFEModel(Node_x, Element, Node_X, Boundary, Element_surface_pressure,
                   None, ElementOrientation, cal_1pk_stress, dtype, device, mode='inverse_mat')
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
#%% estimation of uniform mat
print("run uniform estimation")
raw_mat_uniform=torch.zeros((1, 5), dtype=dtype, device=device, requires_grad=True)
optimizer = LBFGS([raw_mat_uniform], lr=1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
optimizer.set_strong_wolfe(t_max=1, verbose=False)
loss_list=[]
Mat_list=[]
t0=time.time()
#-----------------------------------------------
for iter1 in range(0, arg.max_iter1):
    def closure(loss_fn="MSE"):
        raw_mat_uniform.data.clip_(min=-7, max=7)#avoid vanishing gradient
        Mat=process_raw_mat(raw_mat_uniform)
        LV_model.set_material(Mat)
        out =LV_model.cal_energy_and_force(pressure)
        force_int=out['force_int']
        force_ext=out['force_ext']
        flag_valid=~(torch.isnan(force_int)|torch.isnan(force_ext)|torch.isinf(force_int)|torch.isinf(force_ext))
        loss=loss_function(force_int[flag_valid], force_ext[flag_valid], loss_fn)
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        return loss
    opt_cond=optimizer.step(closure)    
    #
    loss=float(closure(loss_fn="RMSE"))
    loss_list.append(loss)
    t1=time.time()
    #
    if np.isnan(loss) == True or np.isinf(loss) == True:
        print(iter1, loss, t1-t0)
        print('abort: loss is nan or inf')
        sys.exit()
    #
    Mat=process_raw_mat(raw_mat_uniform)
    Mat_list.append(Mat[0].detach().cpu().numpy().reshape(-1).tolist())
    del Mat
    #
    if len(Mat_list) > 100:
        Mat0=np.array(Mat_list[-2])
        Mat1=np.array(Mat_list[-1])
        a=np.abs(Mat0-Mat1).max()
        if a < 1e-10:
            print("a < 1e-10: set opt_cond to True")
            opt_cond=True
    #
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
    #
    if opt_cond == True:
        print("break: opt_cond is True")
        break          
#-----------------------------------------------
raw_mat_uniform=raw_mat_uniform.detach()
Mat_uniform=process_raw_mat(raw_mat_uniform)
Mat_uniform=Mat_uniform.detach().reshape(-1,6)
#%% NN or RawMat to solve for the mat field
if arg.net != "none":
    mat_net=eval(arg.net).to(dtype).to(device)
else:
    RawMat=torch.zeros((Element.shape[0], 5), dtype=dtype, device=device, requires_grad=True)
#%%
def run_mat_net():
    return process_raw_mat(mat_net(NodeInput))
#%%
def cal_mat():
    if arg.net != "none":
        Mat=run_mat_net()
    else:
        Mat=process_raw_mat(RawMat)
    return Mat
#%%
if arg.net == "none":
    RawMat.data-=RawMat.data-raw_mat_uniform.data 
else:    
    print("initilize mat_net with Mat_uniform")
    print(Mat_uniform[0].tolist())
    optimizer = LBFGS(mat_net.parameters(), lr=1, line_search_fn="strong_wolfe",
                      tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
    for iter1 in range(0, 501):
        def closure():
            #Mat=run_mat_net()
            #loss=((Mat-Mat_uniform)**2).mean()
            raw_mat=mat_net(NodeInput)
            loss=((raw_mat-raw_mat_uniform)**2).mean()            
            if loss.requires_grad==True:
                optimizer.zero_grad()
                loss.backward()
            return loss
        optimizer.step(closure)
        if iter1%100==0:
            Mat=run_mat_net().detach()
            Mat_mean=Mat.mean(dim=0).detach().cpu().numpy().tolist()
            print(iter1, Mat_mean)
            del Mat    
#%%
def save_Mat(save_to_p0_or_px):
    with torch.no_grad():
        Mat_element=cal_mat()
        Mat_element=Mat_element.detach()
        Mat_element_true=Mat_true.expand(Element.shape[0],-1)
        Mat_node_true=cal_attribute_on_node(Node_X.shape[0], Element, Mat_element_true)
        Mat_element_true=Mat_element_true.detach().cpu()
        Mat_node=cal_attribute_on_node(Node_X.shape[0], Element, Mat_element)
    Mat_element=Mat_element.detach().cpu()
    Mat_node=Mat_node.detach().cpu()
    Mat_node_true=Mat_node_true.detach().cpu()
    Mesh_mat=PolyhedronMesh()
    if save_to_p0_or_px == 'p0':
        Mesh_mat.node=Node_X.detach().cpu()
    else:
        Mesh_mat.node=Node_x.detach().cpu()
    Mesh_mat.element=Element.detach().cpu()
    Mesh_mat.node_data['Mat_true']=Mat_node_true
    Mesh_mat.node_data['Mat_pred']=Mat_node
    Mesh_mat.node_data['Error']=(Mat_node-Mat_node_true).abs()/(Mat_node_true.mean(dim=0, keepdim=True)+1e-5)
    Mesh_mat.element_data['Mat_true']=Mat_element_true
    Mesh_mat.element_data['Mat_pred']=Mat_element
    Mesh_mat.element_data['Error']=(Mat_element-Mat_element_true).abs()/(Mat_element_true.mean(dim=0, keepdim=True)+1e-5)
    try:
        Mesh_mat.mesh_data['model_state']=mat_net.state_dict()
    except:
        pass
    Mesh_mat.mesh_data['loss']=torch.tensor(loss_list)
    Mesh_mat.mesh_data['memory']=torch.cuda.memory_stats(device=device)
    Mesh_mat.mesh_data['time']=torch.tensor(t_list)
    Mesh_mat.mesh_data['error']=torch.tensor(error_list)
    Mesh_mat.mesh_data['arg']=arg
    if save_to_p0_or_px == 'p0':
        filename=arg.folder_result+'/'+arg.mesh_p0.split('/')[-1]+"_ex_vivo_mat_"+arg.net
    else:
        filename=arg.folder_result+'/'+arg.mesh_px.split('/')[-1]+"_ex_vivo_mat_"+arg.net
    Mesh_mat.save_as_vtk(filename+".vtk")
    Mesh_mat.save_as_torch(filename+".pt")
    print("saved", filename)
#%%
loss_list=[]
error_list=[]
t_list=[]
t0=time.time()
#%% estimation of nonuniform mat
if arg.net !="none":
    optimizer = LBFGS(mat_net.parameters(), lr=1, line_search_fn="strong_wolfe",
                      tolerance_grad=1e-5, tolerance_change=1e-10, history_size=20, max_iter=1)
else:
    optimizer = LBFGS([RawMat], lr=1, line_search_fn="strong_wolfe",
                      tolerance_grad=1e-5, tolerance_change=1e-10, history_size=20, max_iter=1)
#%%
print("run nonuniform estimation")    
for iter2 in range(0, arg.max_iter2):
    def closure(loss_fn="MSE"):
        Mat=cal_mat()
        LV_model.set_material(Mat)
        out =LV_model.cal_energy_and_force(pressure)
        force_int=out['force_int']
        force_ext=out['force_ext']
        flag_valid=~(torch.isnan(force_int)|torch.isnan(force_ext)|torch.isinf(force_int)|torch.isinf(force_ext))
        loss=loss_function(force_int[flag_valid], force_ext[flag_valid], loss_fn)
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        return loss
    opt_cond=optimizer.step(closure)
    #
    loss=float(closure(loss_fn="RMSE"))
    loss_list.append(loss)
    t1=time.time()
    t_list.append(t1-t0)
    #
    if np.isnan(loss) == True or np.isinf(loss) == True:
        print(iter2, loss, t1-t0)
        print('abort: loss is nan or inf')
        sys.exit()
    #
    Mat=cal_mat().detach()  
    Mat_mean=Mat.mean(dim=0).detach().cpu().numpy().tolist()
    Error=(Mat-Mat_true).abs()/(Mat_true.mean(dim=0, keepdim=True)+1e-5)
    Error_mean=Error.mean(dim=0).detach().cpu().numpy().tolist()
    Error_max=Error.max(dim=0)[0].detach().cpu().numpy().tolist()
    Error_min=Error.min(dim=0)[0].detach().cpu().numpy().tolist()
    error_list.append([Error_mean, Error_max, Error_min])
    del Mat
    #
    if (iter2)%100 == 0 or iter2 == arg.max_iter2-1 or opt_cond == True:
        print(iter2, loss, t1-t0)
        print("mat_mean", Mat_mean)
        print("error_mean", Error_mean)
        print("error_max", Error_max)
        print("error_min", Error_min)
        display.clear_output(wait=False)
        fig, ax = plt.subplots()
        ax.plot(np.array(loss_list), 'r')
        ax.set_ylim(0, 0.1)
        ax.grid(True)
        display.display(fig)
        plt.close(fig)
    #
    if opt_cond == True:
        print("opt_cond is True, break")
        break
#%% save
save_Mat(save_to_p0_or_px='px')
save_Mat(save_to_p0_or_px='p0')

