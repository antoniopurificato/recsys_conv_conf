#!/usr/bin/env python
# coding: utf-8

# # Preparation stuff

# ## IMPORTS

# In[ ]:


#Put all imports here
import numpy as np
import matplotlib.pyplot as plt
#from copy import deepcopy
#import pickle
import os
import sys
#import cv2
import torch
import csv
from copy import deepcopy


# In[ ]:


#import setuptools.dist


# ## Define paths

# In[ ]:


#every path should start from the project folder:
project_folder = "../"

#Config folder should contain hyperparameters configurations
cfg_folder = os.path.join(project_folder,"cfg")

#Data folder should contain raw and preprocessed data
data_folder = os.path.join(project_folder,"data")
raw_data_folder = os.path.join(data_folder,"raw")
processed_data_folder = os.path.join(data_folder,"processed")

#Source folder should contain all the (essential) source code
source_folder = os.path.join(project_folder,"src")

#The out folder should contain all outputs: models, results, plots, etc.
out_folder = os.path.join(project_folder,"out")
img_folder = os.path.join(out_folder,"img")


# ## Import own code

# In[ ]:


#To import from src:

#attach the source folder to the start of sys.path
sys.path.insert(0, project_folder)
os.environ['PYTHONPATH'] = project_folder #for raytune workers

#import from src directory
#from src.module import *

import easy_exp, easy_rec, easy_torch #easy_data


# # MAIN

# ## Train

# ### Data

# In[ ]:


cfg = easy_exp.cfg.load_configuration("config_rec")


# In[ ]:


exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)


# In[ ]:


data, maps = easy_rec.preparation.prepare_rec_data(cfg)


# In[ ]:


loaders = easy_rec.preparation.prepare_rec_dataloaders(cfg, data, maps)


# In[ ]:


main_module = easy_rec.preparation.prepare_rec_model(cfg, maps)


# ### Decomposition

# In[ ]:


trainer = easy_torch.preparation.complete_prepare_trainer(cfg, experiment_id, additional_module=easy_rec)#, raytune=raytune)


# In[ ]:


model = easy_torch.preparation.complete_prepare_model(cfg, main_module, additional_module=easy_rec)


# In[ ]:


# Train the model using the prepared trainer, model, and data loaders
easy_torch.process.train_model(trainer, model, loaders, val_key=["val","test"])


# In[ ]:


easy_torch.process.test_model(trainer, model, loaders, test_key=["val","test","train"])


# In[ ]:


# Save experiment
easy_exp.exp.save_experiment(cfg)