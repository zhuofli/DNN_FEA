# DNN-FEA (LV): Finite Element Analysis with Deep Neural Networks

This repository is a research/education codebase for our manuscript:

**"An Integrated DNN-FEA Approach for Inverse Identification of Passive, Heterogeneous Material Parameters of Left Ventricular Myocardium".**

It is forked from the PyTorch-FEA project by Liang et al. and extended/refactored for **left ventricle (LV)** applications, including:
- LV forward inflation with differentiable FEA
- inverse identification of heterogeneous material parameters (DNN-FEA)
- rule-based fiber/material orientation utilities
- HO-style constitutive modeling components used in our LV pipeline

## Upstream Reference (Original PyTorch-FEA)
- Upstream repo: https://github.com/liangbright/pytorch_fea
- Paper (aorta application): https://doi.org/10.1016/j.cmpb.2023.107616
- Preprint: https://www.biorxiv.org/content/10.1101/2023.03.27.533816v1

> Note: The upstream paper demonstrates aorta examples.  
> This repository focuses on **LV** and reorganizes scripts accordingly.

## Repository Structure 
- `torch_fea/` : differentiable FEA core (upstream-based)
- 'mesh/': codes required to process meshes
- `LVFEModel.py` : LV model wrapper used by LV scripts
- `LV_FEA_QN_forward_inflation.py` : LV forward inflation code
- `LV_FEA_inverse_mat_ex_vivo_NN.py` : LV inverse material identification code
- `RBori.py`, `LV_element_orientation.py` : rule-based orientation utilities
- `doc/` : documentation (data format, reproduction notes)
- `examples/` : runnable minimal demos (being cleaned to match LV)

## Quickstart (LV)
### 1) Install dependencies
Please see main/Requirements.txt

### 2) Run LV forward example
Input: undeformed geometry .inp .vtk and .pt files, material parameters
You can adjust the material parameters for inflation process by editing distribution in main/LV_mat_distribution.py
Output: deformed geometry .pt and .vtk files
```bash
python LV_FEA_QN_forward_inflation.py
```

### 3) Run LV inverse example
Input: deformed and undeformed geometry .pt and .vtk files
Output: material parameters saved in .vtk file. You may check the material parameter value and distribution via Paraview.
```bash
python LV_FEA_inverse_mat_ex_vivo_NN.py
```
## Data

Please see examples/lv/README.md for the expected input data format and minimal example guidance.

## Reproducibility

No absolute paths should be required.

All scripts should be runnable by configuring data paths (see examples/lv/README.md).

## 📄 License and Citations

This repository follows the MIT License, as per the upstream [PyTorch-FEA](https://github.com/liangbright/pytorch_fea) project.

Please retain all original LICENSE terms.

If you use this code, please **cite both** our paper and the upstream PyTorch-FEA paper:

### 📚 Our Paper
> DOI remains to be added

### 📚 PyTorch-FEA
> Liang, Liang, et al. "PyTorch-FEA: Autograd-enabled finite element analysis methods with applications for biomechanical analysis of human aorta." Computer methods and programs in biomedicine 238 (2023): 107616. 
> https://doi.org/10.1016/j.cmpb.2023.107616