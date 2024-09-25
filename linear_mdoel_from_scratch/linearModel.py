#!/usr/bin/env python
# coding: utf-8

# In[36]:


import os
from pathlib import Path
import torch, numpy as np, pandas as pd


#For extracting data from kaggle

path = Path('../titanic')
# if not path.exists():
#     import zipfile,kaggle
#     kaggle.api.competition_download_cli(str(path))
#     zipfile.ZipFile(f'{path}.zip').extractall(path)

np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)


# In[37]:


df = pd.read_csv(path/'train.csv')

#converting the missing values to the mode of the column
modes = df.mode().iloc[0]
df.fillna(modes, inplace=True)
df['LogFare'] = np.log(df['Fare']+1)
df = pd.get_dummies(df, columns=["Sex","Pclass","Embarked"])
df.columns
added_cols = ['Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

#converting the boolean fields to int
for col in added_cols:
    df[col] = df[col].astype(int)


t_dep = torch.tensor(df.Survived)
indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare'] + added_cols
#print(df[indep_cols].describe())

t_indep = torch.tensor(df[indep_cols].values, dtype=torch.float)
print(t_indep)


# In[38]:


## Setting up a linear model


torch.manual_seed(442)

n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff)-0.5


#Getting max value ofeach colums so that we can fix the issue in sum of each row in age
vals,indices = t_indep.max(dim=0)
t_indep = t_indep / vals


print("COEFF",coeffs)
print(t_indep*coeffs)
preds = (t_indep*coeffs).sum(axis=1)
print(preds[:10])


# In[39]:


loss = torch.abs(preds-t_dep).mean()
loss


# In[40]:


def calc_preds(coeffs, indeps): return (indeps*coeffs).sum(axis=1)
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()


# In[41]:


coeffs.requires_grad_()


# In[42]:


loss = calc_loss(coeffs, t_indep, t_dep)
loss


# In[43]:


loss.backward()


# In[44]:


coeffs.grad


# In[45]:


loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
coeffs.grad


# In[46]:


loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
with torch.no_grad():
    coeffs.sub_(coeffs.grad * 0.1)
    coeffs.grad.zero_()
    print(calc_loss(coeffs, t_indep, t_dep))


# In[47]:


# from fastai.data.transforms import RandomSplitter
# trn_split,val_split=RandomSplitter(seed=42)(df)


# In[48]:


from fastai.data.transforms import RandomSplitter
trn_split, val_split = RandomSplitter(seed=42)(df)
print(trn_split)


# In[49]:


trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]
len(trn_indep),len(val_indep)


# In[50]:


def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()


# In[51]:


def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep)
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr)
    print(f"{loss:.3f}", end="; ")


# In[ ]:





# In[52]:


def init_coeffs(): return (torch.rand(n_coeff)-0.5).requires_grad_()


# In[53]:


def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    for i in range(epochs): one_epoch(coeffs, lr=lr)
    return coeffs


# In[54]:


coeffs = train_model(18, lr=0.2)


# In[55]:


def show_coeffs(): return dict(zip(indep_cols, coeffs.requires_grad_(False)))
show_coeffs()


# In[56]:


preds = calc_preds(coeffs, val_indep)


# In[57]:


results = val_dep.bool()==(preds>0.5)
results[:16]


# In[58]:


results.float().mean()


# In[59]:


def acc(coeffs): return (val_dep.bool()==(calc_preds(coeffs, val_indep)>0.5)).float().mean()
acc(coeffs)


# In[60]:


def calc_preds(coeffs, indeps): return torch.sigmoid((indeps*coeffs).sum(axis=1))


# In[61]:


coeffs = train_model(lr=100)








