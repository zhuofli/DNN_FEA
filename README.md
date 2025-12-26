# DNN-FEA (LV): Differentiable Finite Element Analysis with Neural Networks

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

## Repository Structure (high level)
- `torch_fea/` : differentiable FEA core (upstream-based)
- `LVFEModel.py` : LV model wrapper used by LV scripts
- `LV_FEA_QN_forward_inflation.py` : LV forward inflation example
- `LV_FEA_inverse_mat_ex_vivo_NN.py` : LV inverse material identification (DNN-FEA)
- `RBori.py`, `LV_element_orientation.py` : rule-based orientation utilities
- `doc/` : documentation (data format, reproduction notes)
- `examples/` : runnable minimal demos (being cleaned to match LV)

## Quickstart (LV)
### 1) Install dependencies
- PyTorch
- PyTorch Geometric
- PyPardiso
- mesh library (required): https://github.com/liangbright/mesh

### 2) Run LV forward example
```bash
python LV_FEA_QN_forward_inflation.py
```
### 3) Run LV inverse example
```bash
python LV_FEA_inverse_mat_ex_vivo_NN.py
```
## Data

Due to size and potential privacy restrictions, full datasets are not hosted in this repository.
Please see doc/data_format.md for the expected input data structure and minimal example guidance.

## Reproducibility

No absolute paths should be required.

All scripts should be runnable by configuring data paths (see doc/data_format.md).

## License / Acknowledgement

This repository follows the upstream licensing terms. Please cite the upstream PyTorch-FEA paper if you use the FEA core.
