#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np


# In[17]:


A = np.array([     
        [2, -1, 1, 1],
        [1, 2, -1, -1],
        [-1, 2, 2, 2],
        [1, -1, 2, 1]    
    ], dtype=np.dtype(float)) 
B = np.array([     
        [2, -1, 1, 1],
        [1, 2, -1, -1],
        [-1, 2, 2, 2],
        [1, -1, 2, 1]    
    ], dtype=np.dtype(float)) 
print(A)


# In[20]:


d = np.linalg.det(A)
e = np.linalg.det(B)
sol = np.linalg.solve(A,B)
print(sol)


# In[28]:


C = np.array([
        [4, -3, 1],
        [2, 1, 3],
        [-1, 2, -5]
    ], dtype=np.dtype(float))

D = np.array([-10, 0, 17], dtype=np.dtype(float))
print(D.reshape((3, 1)))
A_system = np.hstack((C, D.reshape((3, 1))))

print(A_system)




