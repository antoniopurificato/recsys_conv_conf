
#Put all imports here
import numpy as np
import matplotlib.pyplot as plt
#from copy import deepcopy
#import pickle
import os
import sys
#import cv2
import csv
import torch

# ## Define paths

# In[5]:


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

# In[6]:


#To import from src:

#attach the source folder to the start of sys.path
sys.path.insert(0, project_folder)

#import from src directory
# from src import ??? as additional_module
import easy_rec as additional_module #REMOVE THIS LINE IF IMPORTING OWN ADDITIONAL MODULE

import easy_exp, easy_rec, easy_torch #easy_data

# # MAIN

# ## Train

# ### Data

# In[7]:


cfg = easy_exp.cfg.load_configuration("config_rec")

# In[8]:


cfg["data_params"]["data_folder"] = raw_data_folder

# In[9]:


#cfg["data_params"]["test_sizes"] = [cfg["data_params.dataset_params.out_seq_len.val"],cfg["data_params.dataset_params.out_seq_len.test"]]

data, maps = easy_rec.data_generation_utils.preprocess_dataset(**cfg["data_params"])

# In[10]:


#Save user and item mappings
# TODO: check
with open(os.path.join(processed_data_folder,"user_map.csv"), "w") as f_user:
    w = csv.writer(f_user)
    w.writerows(maps['uid'].items())

with open(os.path.join(processed_data_folder,"item_map.csv"), "w") as f_item:
    w = csv.writer(f_item)
    w.writerows(maps['sid'].items())

# In[11]:


datasets = easy_rec.rec_torch.prepare_rec_datasets(data,**cfg["data_params"]["dataset_params"])

# In[12]:


cfg["data_params"]["collator_params"]["num_items"] = np.max(list(maps["sid"].values()))

# In[13]:


collators = easy_rec.rec_torch.prepare_rec_collators(data, **cfg["data_params"]["collator_params"])

# In[14]:


loaders = easy_rec.rec_torch.prepare_rec_data_loaders(datasets, **cfg["model"]["loader_params"], collate_fn=collators)

# In[15]:


cfg["model"]["rec_model"]["num_items"] = np.max(list(maps["sid"].values()))
cfg["model"]["rec_model"]["num_users"] = np.max(list(maps["uid"].values()))
cfg["model"]["rec_model"]["lookback"] = cfg["data_params"]["collator_params"]["lookback"]

# In[16]:


main_module = easy_rec.rec_torch.create_rec_model(**cfg["model"]["rec_model"])

# In[17]:


exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)

# In[18]:


#if exp_found: exit() #TODO: make the notebook/script stop here if the experiment is already found

# In[19]:


trainer_params = easy_torch.preparation.prepare_experiment_id(cfg["model"]["trainer_params"], experiment_id)

# Prepare callbacks and logger using the prepared trainer_params
trainer_params["callbacks"] = easy_torch.preparation.prepare_callbacks(trainer_params)
trainer_params["logger"] = easy_torch.preparation.prepare_logger(trainer_params)

# Prepare the trainer using the prepared trainer_params
trainer = easy_torch.preparation.prepare_trainer(**trainer_params)

model_params = cfg["model"].copy()

model_params["loss"] = easy_torch.preparation.prepare_loss(cfg["model"]["loss"], additional_module.losses)

# Prepare the optimizer using configuration from cfg
model_params["optimizer"] = easy_torch.preparation.prepare_optimizer(**cfg["model"]["optimizer"])

# Prepare the metrics using configuration from cfg
model_params["metrics"] = easy_torch.preparation.prepare_metrics(cfg["model"]["metrics"], additional_module.metrics)

# Create the model using main_module, loss, and optimizer
model = easy_torch.process.create_model(main_module, **model_params)

# In[20]:


# Prepare the emission tracker using configuration from cfg
tracker = easy_torch.preparation.prepare_emission_tracker(**cfg["model"]["emission_tracker"], experiment_id=experiment_id)

# In[21]:


# Prepare the flops profiler using configuration from cfg
profiler = easy_torch.preparation.prepare_flops_profiler(model=model, **cfg["model"]["flops_profiler"], experiment_id=experiment_id)

# ### Train

# In[22]:


# Train the model using the prepared trainer, model, and data loaders
easy_torch.process.train_model(trainer, model, loaders, val_key=["val","test"], tracker=tracker, profiler=profiler) 

# In[ ]:


easy_torch.process.test_model(trainer, model, loaders, tracker=tracker, profiler=profiler)

# In[ ]:


# Save experiment and print the current configuration
#save_experiment_and_print_config(cfg)
easy_exp.exp.save_experiment(cfg)

# Print completion message
print("Execution completed.")
print("######################################################################")
print()
