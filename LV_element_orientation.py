# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:43:23 2024

@author: liang
"""
import torch
#%%
def cal_element_orientation(node, element):
    orientation=torch.eye(3).expand(len(element),3,3)
    return orientation