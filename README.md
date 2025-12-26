# DNN-FEA: PyTorch-based Finite Element Analysis for Cardiac Mechanics

This repository contains a **refactored and extended PyTorch-based Finite Element Analysis (FEA) framework**
adapted for **deep learning–assisted biomechanical modeling**, with a primary focus on **left ventricle (LV) mechanics**.

The codebase is built upon the PyTorch-FEA framework introduced in:

> **PyTorch-FEA: Autograd-enabled Finite Element Analysis Methods with Applications for Biomechanical Analysis of Human Aorta**  
> *Computer Methods and Programs in Biomedicine*, 2023  
> DOI: https://doi.org/10.1016/j.cmpb.2023.107616

The original implementation associated with the paper is available at:  
https://github.com/liangbright/pytorch_fea_paper

This repository **is not a mirror of the original paper code**.  
Instead, it provides a **cleaned, modularized, and application-oriented extension** designed for:
- left ventricle (LV) finite element modeling,
- constitutive parameter learning using neural networks,
- inverse and forward biomechanical analysis with differentiable FEA.

---

## Key Differences from the Original PyTorch-FEA Paper Code

Compared with the original aorta-focused implementation, this repository includes:

- Refactored project structure with clearer separation between:
  - FEA core modules,
  - constitutive models,
  - data handling,
  - experiment scripts.
- Adaptation to **left ventricle geometry and loading conditions**.
- Simplified and cleaned example scripts to support **reproducible forward and inverse analyses**.
- Removal of hard-coded paths and environment-specific dependencies.
- Improved documentation and inline comments for readability and reuse.

---

## Repository Structure

```text
.
├── torch_fea/                # Core FEA and learning modules
│   ├── element/            # Finite element formulation and solvers, support tri3, tri6, hex8, tet10, tet4, wedge6 and quad4
│   ├── material/           # Constitutive model
│   ├── models/             # FE model achievement
│   ├── optimizer/          # LBFGS optimizer in forward and inverse process
│   └── utils/              # Reproducibility utilities
│
├── examples/
│   └── lv/                 # Minimal LV forward and inverse examples
│
├── doc/                    # Methodology and data-format documentation
├── Requirements
└── README.md
```
---

## Minimal Reproducible Example

Forward Analysis

Compute the pressurized LV configuration given material parameters and an unloaded geometry:
main code: LV_FEA_QN_forward_inflation.py
input:
