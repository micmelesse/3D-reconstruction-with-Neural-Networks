
# coding: utf-8

# In[1]:


import os
import sys
import random
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
get_ipython().magic('matplotlib inline')


# In[2]:


meshes=[];
for root, subdirs, files in os.walk('../ShapeNet/'):
    for f in files:
        if (f.endswith('.obj')):
            meshes.append(root+'/'+f)
            


# In[4]:


m= trimesh.load_mesh(meshes[50])
m[0].show()

