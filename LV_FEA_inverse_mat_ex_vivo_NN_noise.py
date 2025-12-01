# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:03:22 2024

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
from torch_fea.utils.functions import cal_attribute_on_node, cal_von_mises_stress
from torch_fea.optimizer.FE_lbfgs_ori import LBFGS
from MatNet import Net0, Net3
from PolyhedronMeshProcessing import PolyhedronMesh, TetrahedronMesh, Tet10Mesh
import time
from LVFEModel import LVFEModel
from LV_mat_distribution import generate_mat_distribution
#%%
all_mat=torch.load('D:/MLFEA/minliang_lv/data/125mat.pt')['mat_str']
matMean=torch.load('D:/MLFEA/minliang_lv/data/125mat.pt')['mean_mat_str']
px_pressure=20
mat_model='GOH_Jv'
#mat_true="1e2, 0, 1, 0, 0, 1e5"; mat_name='1e2'
#mat_true=matMean; mat_name='matMean'
mat_true='generate_mat_distribution(2,arg.mesh_p0)'; mat_name='distribution2'
mesh_p0_str='D:/MLFEA/minliang_lv/data/ori/x4_c3d4_ori'
mesh_px_str=mesh_p0_str+'_inflate_'+mat_model+'('+str(mat_name)+')_p'+str(px_pressure)
folder_result='D:/MLFEA/minliang_lv/data/ori/inverse_mat_ex_vivo_NN_noise'
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--folder_result', default=folder_result, type=str)
parser.add_argument('--mesh_p0', default=mesh_p0_str, type=str)
parser.add_argument('--mesh_px', default=mesh_px_str, type=str)
parser.add_argument('--mesh_input', default=mesh_p0_str, type=str)#template
parser.add_argument('--mat_model', default=mat_model, type=str)
parser.add_argument('--mat_true', default=mat_true, type=str)
parser.add_argument('--pressure', default=px_pressure, type=float)
parser.add_argument('--max_iter1', default=10000, type=int)
parser.add_argument('--max_iter2', default=100000, type=int)
parser.add_argument('--net', default="Net3(3,256,2,1,1,0,5)", type=str)
#parser.add_argument('--net', default="Net0(3,256,4,1,1,0,5)", type=str)
#parser.add_argument('--net', default="none", type=str)
parser.add_argument('--noise_model', default='normal', type=str)#normal or uniform
parser.add_argument('--noise_level', default=0, type=float) # 0~1, noise-to-signal(strain) ratio
parser.add_argument('--random_seed', default=0, type=int)
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
Mesh_x=TetMesh()
Mesh_x.load_from_torch(arg.mesh_px+".pt")
Node_x_clean=Mesh_x.node.to(dtype).to(device)
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
    Mat[:,1]=m1_min+(m1_max-m1_min)*M[:,1]
    Mat[:,2]=m2_min+(m2_max-m2_min)*M[:,2]
    Mat[:,3]=(1/3)*M[:,3]
    Mat[:,4]=(np.pi/2)*M[:,4]
    Mat[:,5]=1e5 # a known constant
    return Mat
#%%
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
#%% initialize u_field
print("initialize u_field: subject to J=1")
Mat_init=Mat=process_raw_mat(torch.zeros((1, 5), dtype=dtype, device=device))
print('Mat_init', Mat_init[0].tolist())
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
del optimizer, loss1, loss2, lossJ, opt_cond, iter1, beta
#%% estimation of uniform mat
print("run uniform estimation")
loss_list=[]
loss1_list=[]
loss2_list=[]
Mat_list=[]
time_list=[]
beta=1
beta_min, beta_max=arg.beta_range
#%%
raw_mat_uniform=torch.zeros((1, 5), dtype=dtype, device=device, requires_grad=True)
optimizer = LBFGS([raw_mat_uniform, u_field], lr=1, line_search_fn="strong_wolfe",
                  tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
optimizer.set_strong_wolfe(t_max=1, verbose=False)
#%%
for iter1 in range(0, arg.max_iter1):
    def closure(return_all=False):
        raw_mat_uniform.data.clip_(min=-7, max=7)#avoid vanishing gradient
        Mat=process_raw_mat(raw_mat_uniform)
        LV_model.set_material(Mat)
        Node_x_pred=Node_X+u_field*mask
        LV_model.set_node_x(Node_x_pred)
        out=LV_model.cal_energy_and_force(pressure)
        force_int=out['force_int']
        force_ext=out['force_ext']
        #flag_valid=~(torch.isnan(force_int)|torch.isnan(force_ext)|torch.isinf(force_int)|torch.isinf(force_ext))
        #loss=cal_loss(force_int[flag_valid], force_ext[flag_valid], "MSE")
        loss1=cal_loss(force_int, force_ext, "MSE")
        loss2=cal_loss(Node_x_pred, Node_x, "MSE")        
        loss=loss1 + loss2*beta
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        if return_all == False:
            return loss
        else:
            return float(loss), float(loss1), float(loss2)
    t0=time.time()
    opt_cond=optimizer.step(closure)
    t1=time.time()
    time_list.append(t1-t0)   
    #
    loss, loss1, loss2=closure(return_all=True)    
    loss_list.append(loss)
    loss1_list.append(loss1)
    loss2_list.append(loss2)
    t1=time.time()
    #
    if np.isnan(loss) == True or np.isinf(loss) == True:
        print(iter1, loss, t1-t0)
        print('abort: loss is nan or inf')
        sys.exit()
    #
    Node_x_pred=Node_X+u_field*mask
    node_error=((Node_x_pred-Node_x_clean)**2).mean().item()
    del Node_x_pred
    #
    if len(loss2_list) >= 100:
        if loss2 > noise_var and loss2_list[-1] >= loss2_list[-10]:
            beta=beta*(loss2+1e-8)/(noise_var+1e-8)    
        elif loss2 < noise_var and loss2_list[-1] <= loss2_list[-10]:
            beta=beta*(loss2+1e-8)/(noise_var+1e-8)
    beta=max(min(beta, beta_max), beta_min)
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
        print("iter1", iter1, "loss", loss, "loss1", loss1, "loss2", loss2, "time", sum(time_list))
        print('noise_var', noise_var, "beta", beta, "node_error", node_error)
        print("Mat:", Mat_list[-1])
        #display.clear_output(wait=False)
        #fig, ax = plt.subplots()
        #ax.plot(np.array(loss_list)/max(loss_list), 'r')
        #ax.set_ylim(0, 1)
        #ax.grid(True)
        #display.display(fig)
        #plt.close(fig)
    #
    if opt_cond == True:
        print("break: opt_cond is True")
        break          
#-----------------------------------------------
raw_mat_uniform=raw_mat_uniform.detach()
Mat_uniform=process_raw_mat(raw_mat_uniform)
Mat_uniform=Mat_uniform.detach().reshape(-1,6)
#%%
del optimizer, loss_list
if arg.max_iter1 > 0:
    del loss, opt_cond, iter1
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
    RawMat.data-=RawMat.data.data-raw_mat_uniform.data 
else:    
    print("initilize mat_net with Mat_uniform")
    print(Mat_uniform[0].tolist())
    optimizer = LBFGS(mat_net.parameters(), lr=1, line_search_fn="strong_wolfe",
                      tolerance_grad=1e-10, tolerance_change=1e-10, history_size=20, max_iter=1)
    for iter1 in range(0, 1000):
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
    Mesh_mat.mesh_data['loss1']=torch.tensor(loss1_list)
    Mesh_mat.mesh_data['loss2']=torch.tensor(loss2_list)
    Mesh_mat.mesh_data['memory']=torch.cuda.memory_stats(device=device)
    Mesh_mat.mesh_data['time']=torch.tensor(time_list)
    Mesh_mat.mesh_data['error']=torch.tensor(error_list)
    Mesh_mat.mesh_data['arg']=arg
    if save_to_p0_or_px == 'p0':
        filename=arg.folder_result+'/'+arg.mesh_p0.split('/')[-1]+"_ex_vivo_mat_"+arg.net
    else:
        filename=arg.folder_result+'/'+arg.mesh_px.split('/')[-1]+"_ex_vivo_mat_"+arg.net
    filename=filename+"_"+arg.noise_model+'('+str(arg.noise_level)+')'+"_s"+str(arg.random_seed)
    Mesh_mat.save_as_vtk(filename+".vtk")
    Mesh_mat.save_as_torch(filename+".pt")
    print("saved", filename)
#%%
print("run nonuniform estimation")
loss1_list=[]
loss2_list=[]
error_list=[]
time_list=[]
t0=time.time()
beta=1
beta_min, beta_max=arg.beta_range
#%% estimation of nonuniform mat
if arg.net !="none":
    optimizer = LBFGS(list(mat_net.parameters())+[u_field], lr=1, line_search_fn="strong_wolfe",
                      tolerance_grad=1e-5, tolerance_change=1e-10, history_size=20, max_iter=1)
else:
    optimizer = LBFGS([RawMat, u_field], lr=1, line_search_fn="strong_wolfe",
                      tolerance_grad=1e-5, tolerance_change=1e-10, history_size=20, max_iter=1)
#%%
for iter2 in range(0, arg.max_iter2):
    def closure(return_all=False):
        Mat=cal_mat()
        LV_model.set_material(Mat)
        Node_x_pred=Node_X+u_field*mask
        LV_model.set_node_x(Node_x_pred)
        out=LV_model.cal_energy_and_force(pressure)
        force_int=out['force_int']
        force_ext=out['force_ext']
        #flag_valid=~(torch.isnan(force_int)|torch.isnan(force_ext)|torch.isinf(force_int)|torch.isinf(force_ext))
        #loss=cal_loss(force_int[flag_valid], force_ext[flag_valid], loss_fn)
        loss1=cal_loss(force_int, force_ext, "MSE")
        loss2=cal_loss(Node_x_pred, Node_x, "MSE")        
        loss=loss1 + loss2*beta
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        if return_all == False:
            return loss
        else:
            return float(loss), float(loss1), float(loss2)
    t0=time.time()
    opt_cond=optimizer.step(closure)
    t1=time.time()
    time_list.append(t1-t0)
    #
    loss, loss1, loss2=closure(return_all=True)
    loss1_list.append(loss1)
    loss2_list.append(loss2)
    #
    if np.isnan(loss) == True or np.isinf(loss) == True:
        print(iter2, loss, t1-t0)
        print('abort: loss is nan or inf')
        sys.exit()
    #
    Node_x_pred=Node_X+u_field*mask
    node_error=((Node_x_pred-Node_x_clean)**2).mean().item()
    del Node_x_pred
    #
    if len(loss2_list) >= 100:
        if loss2 > noise_var and loss2_list[-1] >= loss2_list[-10]:
            beta=beta*(loss2+1e-8)/(noise_var+1e-8)    
        elif loss2 < noise_var and loss2_list[-1] <= loss2_list[-10]:
            beta=beta*(loss2+1e-8)/(noise_var+1e-8)
    beta=max(min(beta, beta_max), beta_min)
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
        print("iter2", iter2, "loss", loss, "loss1", loss1, "loss2", loss2, "time", sum(time_list))
        print('noise_var', noise_var, "beta", beta, "node_error", node_error)
        print("mat_mean", Mat_mean)
        print("error_mean", Error_mean)
        print("error_max", Error_max)
        print("error_min", Error_min)
        #display.clear_output(wait=False)
        #fig, ax = plt.subplots()
        #ax.plot(np.array(loss1_list), 'r')
        #ax.set_ylim(0, 0.1)
        #ax.grid(True)
        #display.display(fig)
        #plt.close(fig)
    #
    if opt_cond == True:
        print("opt_cond is True, break")
        break
#%% save
save_Mat(save_to_p0_or_px='px')
save_Mat(save_to_p0_or_px='p0')

