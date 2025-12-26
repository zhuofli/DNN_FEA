# Data Format (LV)

This repo expects LV datasets organized as:

data/lv_case_001/mesh.inp                # tetrahedral mesh (abaqus .inp file)


## Notes
- Units: geometry in mm, pressure in kPa
- Orientation should be saved in .inp file

## Minimal Example
use a downsampled example in main/example to reproduce the result.

### 1) run
```bash
python preprocess_data.py
```
to generate the mesh.vtk and mesh.pt
### 2) run
```bash
python LV_FEA_QN_forward_inflation.py
```
to generate the deformed geometry
### 3) run
```bash
python LV_FEA_inverse_mat_ex_vivo_NN.py
```
