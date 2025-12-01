import sys
sys.path.append('D:/MLFEA/code/mesh')
import numpy as np
import torch
import time
from PolygonMeshProcessing import TriangleMesh, PolygonMesh, SmoothAndProject, SimpleSmootherForMesh
from LoadMeshFromINPFile import read_abaqus_inp
from PolyhedronMeshProcessing import PolyhedronMesh, TetrahedronMesh, Tet10Mesh, ExtractSurface
from SavePointAsVTKFile import save_point_as_vtk
from collections import defaultdict
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
#%%
filename='C:/Users/zhuofli/OneDrive - Texas Tech University/Data_Zhuofan/NNFEA/data/data/Zhuofan result/RBori/x2_c3d10_testori'
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

#%% LV_solid
if 'c3d4' in filename:
    TetMesh=TetrahedronMesh
elif 'c3d10' in filename:
    TetMesh=Tet10Mesh
else:
    raise ValueError("only support c3d4 and c3d10")
LV_solid=TetMesh(node, element)
#%% read endo surface info
#---------------------------------------------
surface_name_S1='_LV_SURF_S1_1'
surface_name_S2='_LV_SURF_S2_1'
surface_name_S3='_LV_SURF_S3_1'
surface_name_S4='_LV_SURF_S4_1'
#---------------------------------------------
'''Element_surface_pressure_tri3=[]
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
    Boundary.extend(value) '''
#%% read all the outsurface info
outsurface_name_S1='_outsurface_S1'
outsurface_name_S2='_outsurface_S2'
outsurface_name_S3='_outsurface_S3'
outsurface_name_S4='_outsurface_S4'
#---------------------------------------------
'''Element_outsurface_pressure_tri3=[]
Element_outsurface_pressure_tri6=[]
Element_outsurface_pressure_poly=[]
for e_idx in element_set[outsurface_name_S1]:
    elm=element[e_idx]
    if element_type[e_idx] == 'C3D4': #face1: 1,2,3
        Element_outsurface_pressure_tri3.append([elm[0], elm[1], elm[2]])
    elif element_type[e_idx] == 'C3D10': #face1: 1,5,2,6,3,7
        Element_outsurface_pressure_tri3.append([elm[0], elm[1], elm[2]])
        Element_outsurface_pressure_tri6.append([elm[0], elm[1], elm[2], elm[4], elm[5], elm[6]])
        Element_outsurface_pressure_poly.append([elm[0], elm[4], elm[1], elm[5], elm[2], elm[6]])
for e_idx in element_set[outsurface_name_S2]:
    elm=element[e_idx]
    if element_type[e_idx] == 'C3D4': #face2: 1,4,2 
        Element_outsurface_pressure_tri3.append([elm[0], elm[3], elm[1]])
    elif element_type[e_idx] == 'C3D10': #face2: 1,8,4,9,2,5
        Element_outsurface_pressure_tri3.append([elm[0], elm[3], elm[1]])
        Element_outsurface_pressure_tri6.append([elm[0], elm[3], elm[1], elm[7], elm[8], elm[4]])
        Element_outsurface_pressure_poly.append([elm[0], elm[7], elm[3], elm[8], elm[1], elm[4]])
for e_idx in element_set[outsurface_name_S3]:
    elm=element[e_idx]
    if element_type[e_idx] == 'C3D4': #face3: 2,4,3
        Element_outsurface_pressure_tri3.append([elm[1], elm[3], elm[2]])
    elif element_type[e_idx] == 'C3D10': #face3: 2,9,4,10,3,6
        Element_outsurface_pressure_tri3.append([elm[1], elm[3], elm[2]])    
        Element_outsurface_pressure_tri6.append([elm[1], elm[3], elm[2], elm[8], elm[9], elm[5]])        
        Element_outsurface_pressure_poly.append([elm[1], elm[8], elm[3], elm[9], elm[2], elm[5]])        
for e_idx in element_set[outsurface_name_S4]:
    elm=element[e_idx]
    if element_type[e_idx] == 'C3D4': #face4: 1,3,4
        Element_outsurface_pressure_tri3.append([elm[0], elm[2], elm[3]])
    elif element_type[e_idx] == 'C3D10': #face4: 1,7,3,10,4,8
        Element_outsurface_pressure_tri3.append([elm[0], elm[2], elm[3]])   
        Element_outsurface_pressure_tri6.append([elm[0], elm[2], elm[3], elm[6], elm[9], elm[7]])
        Element_outsurface_pressure_poly.append([elm[0], elm[6], elm[2], elm[9], elm[3], elm[7]])
if element_type[0] == 'C3D4':
    Element_outsurface_pressure=Element_outsurface_pressure_tri3
    Element_outsurface_pressure_poly=Element_outsurface_pressure_tri3
else:
    Element_outsurface_pressure=Element_outsurface_pressure_tri6
Boundary=[]
for key, value in node_set.items():   
    Boundary.extend(value) '''
#%%
#LV_inner_surface=PolygonMesh(node, Element_surface_pressure_poly)
#LV_inner_surface.save_as_vtk(filename+"_inner_surface_poly.vtk")
#LV_inner_surface_tri3=TriangleMesh(node, Element_surface_pressure_tri3)
#LV_inner_surface_tri3.save_as_vtk(filename+"_inner_surface_tri3.vtk")
#LV_inner_surface_tri3.update_node_normal()
#LV_inner_surface_tri3.node=LV_inner_surface_tri3.node+LV_inner_surface_tri3.node_normal
#LV_inner_surface_tri3.save_as_vtk(filename+"_inner_surface_tri3_offset.vtk")
#%%

# =======================================
#  Endocardial elements   (endo)
# =======================================
endo_elements = []
for name in [surface_name_S1, surface_name_S2, surface_name_S3, surface_name_S4]:
    endo_elements.extend(element_set[name])

endo_elements = sorted(list(set(endo_elements)))
print("Number of endocardial elements (inner surface):", len(endo_elements))
print("Example endo element labels:", endo_elements[:20])
#np.savetxt(filename + "_endo_element_labels.txt", endo_elements, fmt='%d')


# =======================================
#  Outsurface elements  (full boundary)
# =======================================
outer_elements = []
for name in [outsurface_name_S1, outsurface_name_S2, outsurface_name_S3, outsurface_name_S4]:
    outer_elements.extend(element_set[name])

outer_elements = sorted(list(set(outer_elements)))
print("Number of outer-boundary elements (full LV surface):", len(outer_elements))
print("Example outer-boundary elements:", outer_elements[:20])
#np.savetxt(filename + "_outer_element_labels.txt", outer_elements, fmt='%d')


# LV_BC 是节点集合（node labels）
base_nodes = node_set['LV_BC']   # 你的代码里 node_set 保存节点集
print("Base nodes (valve rings) =", len(base_nodes))

base_elements = []

for e in outer_elements:        # outer_elements 是 element 下标（0-based）
    el_nodes = element[e]          # 原 mesh 节点 label，例如 [105,856,244,920]

    # 若该单元任何节点属于 base_nodes，则这个单元是 base element
    if any(n in base_nodes for n in el_nodes):
        base_elements.append(e)

base_elements = sorted(set(base_elements))
print("Base elements (outer ∩ LV_BC-adjacent) =", len(base_elements))
print("Example base elements:", base_elements[:20])

# =======================================
#  Epicardial elements  (outer − endo)
# =======================================
epi_elements = sorted(set(outer_elements) - set(endo_elements) - set(base_elements))

print("Epicardial (true) elements =", len(epi_elements))
print("Example epicardial elements:", epi_elements[:20])
#np.savetxt(filename + "_epi_element_labels.txt", epi_elements, fmt='%d')

#%%
# =======================================
#  assign node label to node index
# =======================================
elem = np.array(element)        # shape = (N_elems, 4)
# 所有出现过的 node label（一般就是1..N，但稳妥起见）
all_node_labels = sorted({nid for el in elem for nid in el})
label2idx = {lab: i for i, lab in enumerate(all_node_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}

# 把 element 换成 node index 版本，后面 Laplace / ∇φ 全用这个
elem_idx = np.array([[label2idx[nid] for nid in el] for el in elem], dtype=int)
N_nodes = len(all_node_labels)
N_elems = elem_idx.shape[0]
# For C3D4: elem_idx[e] = corner nodes directly
# For C3D10: elem_idx[e][0:4] are the 4 corner nodes
corner_idx = elem_idx[:, :4] if element_type[0] == 'C3D10' else elem_idx
print("Total nodes:", N_nodes)
print("Total elements:", N_elems)

# base_nodes 原来是 node label，把它也变成 index
base_nodes_idx = set(label2idx[n] for n in base_nodes)

# endocardium nodes (Dirichlet φ=0)
endo_nodes_idx = set()
for e in endo_elements:          # e 是 element 的 0-based index
    for nid in elem[e]:
        endo_nodes_idx.add(label2idx[nid])

# epicardium nodes (Dirichlet φ=1)
epi_nodes_idx = set()
for e in epi_elements:
    for nid in elem[e]:
        epi_nodes_idx.add(label2idx[nid])

print("endo_nodes:", len(endo_nodes_idx))
print("epi_nodes :", len(epi_nodes_idx))
print("base_nodes:", len(base_nodes_idx))
#%%
# 邻接表（节点之间有 edge 就算邻居）
neighbors = [set() for _ in range(N_nodes)]
for e, etype in zip(elem_idx, element_type):

    # --- 取出 corner nodes ---
    if "10" in etype:     # C3D10
        c = e[:4]         # 只用 corner
    else:                 # C3D4
        c = e             # 全部 4 个节点

    # --- 组合 6 条边 (i,j,k,l) ---
    i, j, k, l = c
    edges = [(i,j), (i,k), (i,l),
             (j,k), (j,l),
             (k,l)]

    for a,b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)


# A φ = b
A = lil_matrix((N_nodes, N_nodes), dtype=float)
b = np.zeros(N_nodes, dtype=float)

for i in range(N_nodes):
    if i in endo_nodes_idx:
        # endocardium: φ = 0
        A[i,i] = 1.0
        b[i] = 0.0
    elif i in epi_nodes_idx:
        # epicardium: φ = 1
        A[i,i] = 1.0
        b[i] = 1.0
    else:
        # interior + base → 图拉普拉斯
        nbrs = list(neighbors[i])
        deg = len(nbrs)
        if deg == 0:
            A[i,i] = 1.0
            b[i] = 0.0
        else:
            A[i,i] = deg
            for j in nbrs:
                A[i,j] = -1.0
            # b[i] 默认是 0

print("Solving Laplace system for φ...")
phi = spsolve(A.tocsr(), b)   # φ: 每个 node index 的值
for i in base_nodes_idx:
    phi[i] = 1.0   # treat base as epicardium
print("φ solved. min={:.4f}, max={:.4f}".format(phi.min(), phi.max()))
#%%
# ============================================================
# ===   0. 估计长轴（沿用原版）                         ===
# ============================================================
center_endo = node[list(endo_nodes_idx)].mean(axis=0)
center_epi  = node[list(epi_nodes_idx)].mean(axis=0)
long_axis = center_epi - center_endo
long_axis /= np.linalg.norm(long_axis)
print("Estimated long-axis direction =", long_axis)

# ============================================================
# ===   1. 计算单元级 raw ∇φ（不归一化）                ===
# ============================================================
grad_phi_raw = np.zeros((N_elems, 3))

for e_id in range(N_elems):
    # 取 4 个角点节点
    if "10" in element_type[e_id].upper():
        ids = elem_idx[e_id, :4]
    else:
        ids = elem_idx[e_id]
    
    X = node[ids, :]
    phi_e = phi[ids]

    M = np.column_stack((np.ones(4), X))  # 4×4

    a_b = np.linalg.solve(M, phi_e)       # scalar + gradient
    grad_phi_raw[e_id] = a_b[1:]          # gradient only


# ============================================================
# ===   2. 将单元梯度平均到节点（关键修复）              ===
# ============================================================
node_grad = np.zeros((N_nodes,3))
node_count = np.zeros(N_nodes)

for e_id in range(N_elems):
    ids = elem_idx[e_id]
    for nid in ids:
        node_grad[nid] += grad_phi_raw[e_id]
        node_count[nid] += 1

node_grad /= node_count[:,None]


# ============================================================
# ===   3. 得到 node-based n0（归一化）                   ===
# ============================================================
node_n0 = node_grad.copy()
norms = np.linalg.norm(node_n0, axis=1)

good = norms > 1e-12
avg_n0 = node_n0[good].mean(axis=0)
avg_n0 /= np.linalg.norm(avg_n0)

# normalize good nodes
node_n0[good] /= norms[good][:,None]
# assign fallback to bad nodes
node_n0[~good] = avg_n0


# ============================================================
# ===   4. 将节点 n0 平均到单元                        ===
# ============================================================
n0 = np.zeros((N_elems,3))
for e_id in range(N_elems):
    ids = elem_idx[e_id]
    n = node_n0[ids].mean(axis=0)
    n0[e_id] = n / np.linalg.norm(n)


# ============================================================
# ===   5. 使用 inp 文件给出的 local orientation FSN       ===
# ============================================================
# element_orientation shape = (N_elems, 3, 3)
# columns are: fiber, sheet, normal
f0_local = element_orientation[:,:,0]
s0_local = element_orientation[:,:,1]
n0_local = element_orientation[:,:,2]

# 确保都是 unit vectors
f0_local /= np.linalg.norm(f0_local, axis=1)[:,None]
s0_local /= np.linalg.norm(s0_local, axis=1)[:,None]
n0_local /= np.linalg.norm(n0_local, axis=1)[:,None]

# ============================================================
# ===   6. thickness coordinate φ(d) (沿用原版 Laplace φ)     ===
# ============================================================
phi_elem = np.zeros(N_elems)
for e_id in range(N_elems):
    ids = elem_idx[e_id]
    phi_elem[e_id] = phi[ids].mean()   # ∈ [0,1]

# ============================================================
# ===   7. Helix α(d) and transverse β(d)                    ===
# ============================================================
# 你给的参数：
# αendo = 40°, αepi = -50°
# βendo = -65°, βepi = 25°
alpha = np.deg2rad(40.0)*(1-phi_elem) + np.deg2rad(-50.0)*phi_elem
beta  = np.deg2rad(-65.0)*(1-phi_elem) + np.deg2rad(25.0)*phi_elem

# =# ============================================================
# ===   8. 在 local FSN 基底上旋转：得到最终 F,S,N           ===
# ============================================================

# 先把 base element 做成 boolean mask
is_base = np.zeros(N_elems, dtype=bool)
is_base[base_elements] = True

F = np.zeros((N_elems,3))
S = np.zeros((N_elems,3))
T = np.zeros((N_elems,3))

def rodrigues_rotate(v, k, theta):
    """ rotate vector v around axis k by angle theta """
    k = k / np.linalg.norm(k)
    v_rot = (v*np.cos(theta)
             + np.cross(k, v)*np.sin(theta)
             + k*np.dot(k, v)*(1-np.cos(theta)))
    return v_rot

for e_id in range(N_elems):

    # ------- 情况 1：base elements → 不做 α/β 旋转 -------
    if is_base[e_id]:
        F[e_id] = f0_local[e_id]
        S[e_id] = s0_local[e_id]
        T[e_id] = n0_local[e_id]
        continue

    # ------- 情况 2：非 base elements → 做 α(d), β(d) -------
    f0 = f0_local[e_id]
    s0 = s0_local[e_id]
    n0 = n0_local[e_id]

    # 第一次旋转：helix angle α(d)，在 (f0,s0) 平面内，绕 n0
    a = alpha[e_id]
    f_helix = f0*np.cos(a) + s0*np.sin(a)
    f_helix /= np.linalg.norm(f_helix)

    # 第二次旋转：transverse angle β(d)，绕 fiber
    b = beta[e_id]
    s_rot = rodrigues_rotate(s0, f_helix, b)
    s_rot /= np.linalg.norm(s_rot)

    # normal
    t = np.cross(f_helix, s_rot)
    t /= np.linalg.norm(t)

    F[e_id] = f_helix
    S[e_id] = s_rot
    T[e_id] = t



#%%check orientation continous or not
def build_element_neighbors(elem_idx, element_type):
    """
    支持 C3D4 和 C3D10，两者都只使用 4 个 corner node 构建面。
    elem_idx: (N_elems, nnode_per_element)
    """
    M = elem_idx.shape[0]

    # 获取 corner nodes（C3D4: 4, C3D10: 前 4 个）
    if "10" in element_type[0].upper():
        corners = elem_idx[:, :4]
    else:
        corners = elem_idx  # C3D4

    face_to_elems = defaultdict(list)
    neighbors = [[] for _ in range(M)]

    for e in range(M):
        c = corners[e]
        # 四个三角面
        faces = [
            tuple(sorted((c[0], c[1], c[2]))),
            tuple(sorted((c[0], c[1], c[3]))),
            tuple(sorted((c[0], c[2], c[3]))),
            tuple(sorted((c[1], c[2], c[3]))),
        ]
        for f in faces:
            face_to_elems[f].append(e)

    for f, es in face_to_elems.items():
        if len(es) == 2:
            a, b = es
            neighbors[a].append(b)
            neighbors[b].append(a)

    return neighbors



def detect_flips(F, S, T, neighbors):
    M = len(F)
    fiber_flips = 0
    sheet_flips = 0
    normal_flips = 0

    flip_map = np.zeros(M)

    for e in range(M):
        f = F[e]; s = S[e]; t = T[e]
        for nb in neighbors[e]:
            if np.dot(f, F[nb]) < 0:
                fiber_flips += 1
                flip_map[e] += 1
            if np.dot(s, S[nb]) < 0:
                sheet_flips += 1
                flip_map[e] += 1
            if np.dot(t, T[nb]) < 0:
                normal_flips += 1
                flip_map[e] += 1

    print("Fiber flips   =", fiber_flips)
    print("Sheet flips   =", sheet_flips)
    print("Normal flips  =", normal_flips)

    return flip_map

def smooth_vector_field(V, neighbors, iters=10, lam=0.5):
    M = len(V)
    Vnew = V.copy()

    for _ in range(iters):
        Vtemp = Vnew.copy()
        for e in range(M):
            nb = neighbors[e]
            if len(nb) == 0: 
                continue
            avg = np.mean(Vnew[nb], axis=0)
            Vtemp[e] = (1-lam)*Vnew[e] + lam*avg
        # normalize
        Vtemp /= np.linalg.norm(Vtemp, axis=1)[:,None]
        Vnew = Vtemp

    return Vnew

def smooth_orientation(F, S, T, neighbors,
                       iter_f=15, iter_s=10, lambda_f=0.4, lambda_s=0.4):

    # 1) Smooth fiber field F
    F_s = smooth_vector_field(F, neighbors, iters=iter_f, lam=lambda_f)

    # 2) Recompute sheet direction to maintain orthogonality
    #    Option 1: keep original sheet direction projection
    S_proj = S - np.sum(S * F_s, axis=1)[:,None] * F_s
    S_proj /= np.linalg.norm(S_proj, axis=1)[:,None]

    # Optionally smooth sheet again
    S_s = smooth_vector_field(S_proj, neighbors, iters=iter_s, lam=lambda_s)

    # 3) Compute normal = F × S (ensure right-hand system)
    T_s = np.cross(F_s, S_s)
    T_s /= np.linalg.norm(T_s, axis=1)[:,None]

    return F_s, S_s, T_s

neighbors = build_element_neighbors(elem_idx, element_type)

# detect discontinuity
flip_map = detect_flips(F, S, T, neighbors)
print("Total flipped elements =", np.sum(flip_map>0))

# smoothing
F_s, S_s, T_s = smooth_orientation(F, S, T, neighbors)

# re-check flips
flip_map2 = detect_flips(F_s, S_s, T_s, neighbors)
print("Flips after smoothing =", np.sum(flip_map2>0))

# save orientation
ori = np.hstack((F_s, S_s, T_s))
np.savetxt(filename+"_orientation_smoothed.txt", ori)
print("Saved smoothed orientation.")

#%%
# ============================================================
# ===  9. Output VTK for ParaView visualization               ===
# ============================================================
vtk_filename = filename+"_lv_orientation.vtk"

with open(vtk_filename, 'w') as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("LV rule-based fiber orientation\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")

    # ---------- POINTS ----------
    f.write(f"POINTS {N_nodes} float\n")
    for i in range(N_nodes):
        x, y, z = node[i]
        f.write(f"{x:.6e} {y:.6e} {z:.6e}\n")

    # ---------- CELLS ----------
    # 每个 cell 需要: (1 + nnode)
    # tetra:       5
    # quadratic:   11
    total_size = 0
    for et in element_type:
        if "10" in et.upper():
            total_size += 1 + 4   # 只写 corner nodes
        else:
            total_size += 1 + 4

    f.write(f"\nCELLS {N_elems} {total_size}\n")

    for e_id in range(N_elems):
        et = element_type[e_id].upper()
        conn = elem_idx[e_id]

        if "10" in et:
            # C3D10: 使用前 4 个 corner nodes
            corners = conn[:4]
        else:
            corners = conn  # C3D4 已经是 4 节点

        i, j, k, l = corners
        f.write(f"4 {i} {j} {k} {l}\n")

    # ---------- CELL_TYPES ----------
    f.write(f"\nCELL_TYPES {N_elems}\n")
    for et in element_type:
            f.write("10\n")   # VTK_TETRA

    # ---------- CELL_DATA ----------
    f.write(f"\nCELL_DATA {N_elems}\n")

    # Fiber
    f.write("VECTORS fiber float\n")
    for v in F:
        f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")

    # Sheet
    f.write("\nVECTORS sheet float\n")
    for v in S:
        f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")

    # Normal
    f.write("\nVECTORS normal float\n")
    for v in T:
        f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")


print("VTK written:", vtk_filename)
# ===============================================================
# 10. Save INP in Distribution Format (fully compatible)
# ===============================================================

out_inp = filename + "_with_ori.inp"

with open(out_inp, "w") as f:
    f.write("*Heading\n")
    f.write("** Orientation assigned using RBori_step1.py\n\n")
    
    # Nodes
    f.write("*Node\n")
    for i, xyz in enumerate(node, start=1):
        f.write(f"{i}, {xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}\n")

    # Elements
    f.write("\n*Element, type={}\n".format(element_type[0]))
    for eid, conn in enumerate(element, start=1):
        line = f"{eid}, " + ", ".join(str(n) for n in conn)
        f.write(line + "\n")

    # Node sets
    for name, nds in node_set.items():
        f.write(f"\n*Nset, nset={name}\n")
        for i in range(0, len(nds), 16):
            f.write(", ".join(str(x) for x in nds[i:i+16]) + "\n")

    # Element sets
    for name, els in element_set.items():
        f.write(f"\n*Elset, elset={name}\n")
        for i in range(0, len(els), 16):
            f.write(", ".join(str(x+1) for x in els[i:i+16]) + "\n")

    # -------------------------------
    # Distribution Table
    # -------------------------------
    f.write("\n*Distribution Table, name=ORI_TABLE\n")
    f.write("COORD3, COORD3, COORD3, COORD3, COORD3, COORD3, COORD3, COORD3, COORD3\n")

    # -------------------------------
    # Element Orientation Distribution
    # -------------------------------
    f.write("\n*Distribution, name=ORI, location=element, table=ORI_TABLE\n")

    for eid in range(N_elems):
        f0 = F[eid]
        s0 = S[eid]
        n0 = T[eid]
        vals = list(f0) + list(s0) + list(n0)
        f.write(f"{eid+1}, " + ", ".join(f"{v:.6f}" for v in vals) + "\n")

    # Material section (generic)
    f.write("\n*Solid Section, elset=ALL, material=MAT1, orientation=ORI\n")
    f.write("\n*Material, name=MAT1\n*Elastic\n1E6, 0.45\n")

print("Wrote orientation distribution INP to:", out_inp)
