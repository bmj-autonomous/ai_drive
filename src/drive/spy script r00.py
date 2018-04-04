#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 11:44:31 2018

@author: batman
"""

#%% Setup

import os, re, logging, json
from pprint import pprint
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import cv2 #conda install --channel https://conda.anaconda.org/menpo opencv3
from datetime import datetime
import random
import time
import sys
from collections import defaultdict

print_imports()


mod_path = r"/home/batman/git/ai_drive/src"
sys.path.append(mod_path)
logging.debug("ADDED TO PATH: ".format(mod_path))
import drive.analysis_offline as analysis
import drive.my_generators as my_generators
import drive.my_plotting as my_plotting

print(analysis)

#%% Paths 

# Project path
this_project_path = r"/media/batman/USB STICK"
project_name = r'catdog5'
path_root_project = os.path.join(this_project_path,project_name)
assert os.path.exists(path_root_project)

# Full data
path_data_root = r"/home/batman/Dropbox/DATA/cats_dogs_all_test_split"
path_test = os.path.join(path_data_root, 'my_test')

# Test data path
path_cats = os.path.join(path_test,'cats')
path_dogs = os.path.join(path_test,'dogs')

### Constants
IMG_SIZE = 150
layer_funcs = analysis.LAYER_FUNCS

### Folders
run_folders = [dir for dir in os.listdir(path_root_project) if re.match('run',dir)]
run_folders.sort()

#%% Next
run_list = list()
# Loop run folders
for rfolder in run_folders:
    
    summary=dict()
    
    this_run_path = os.path.join(path_root_project,rfolder)
    logging.debug('**** RUN {} ****'.format(rfolder))
    
    summary['run'] = rfolder
    
    runnum = re.findall(r'\d+', summary['run'])[0]
    summary['runi'] = int(runnum)
    
    
    
    ###### Log file ######
    log = analysis.get_log_file(this_run_path)
    
    #print('start;',log['start'])
    summary['start'] = log['start'].__str__()
    #print('elapsed; {:.1f}'.format(log['elapsed'].seconds/60))
    summary['elapsed'] = log['elapsed'].seconds/60
    #print('generator;',log['generator'])
    summary['generator'] = log['generator']
    del log
    
    ###### Weights ######
    wts = analysis.get_weights(this_run_path)
    best_wt = wts[-1] # BEST weight (last weight)
    del wts # Save space
    
    ###### Architecture ######
    model = analysis.load_model(this_run_path)
    arch_dict = analysis.read_model_json(this_run_path)
    
    #model.summary()
    parameter_counts = analysis.count_params(model)
    summary['param_counts'] = parameter_counts
    #print(analysis.count_params(model))
    
    ##### Initialize model, reload weights, best #####
    #if wts:
        #print(best_wt)
    model.load_weights(best_wt['path'])
    logging.debug("Loaded weights into model")
    #del model

    ###### Loop layers ######
    #layer_list = {i:layer for i,layer in enumerate(arch_dict['config'])}
    #layer_list = {i:layer for i,layer in enumerate(arch_dict['config'])}
    
    for i,layer in enumerate(arch_dict['config']):
        if layer['class_name'] == 'Dropout':
            #print(layer['config']['rate'])
            summary['Dropout'] = layer['config']['rate']
            
    layer_summary = defaultdict(int)
    for layer in arch_dict['config']:
        layer_summary[layer['class_name']] += 1
    summary['layers'] = layer_summary
    
    ###### History ######
    path_hist = analysis.get_history(this_run_path)    
    with open(path_hist) as hist_file:
        hist_dict = json.load(hist_file)
    summary['hist_dict'] = hist_dict
    
    ###### Done ######
    run_list.append(summary)
    #break
#%% Plot
    
#import matplotlib as mpl
#mpl.use('Agg')
#this_run = run_list[12]

for this_run in run_list:
    assert this_run['layers']['Conv2D'] == this_run['layers']['MaxPooling2D']
    title_str = '{} - {}xConvPool, {} dropout, {:0.1f}M parameters'.format(
            this_run['runi'],
            this_run['layers']['Conv2D'],
            this_run['Dropout'],
            this_run['param_counts']['Total']/1000000)
    
    #title_str.append()
    
    this_plot = my_plotting.plot_hist_dict(this_run['hist_dict'],title_str)
    save_path = os.path.join(path_root_project,title_str+'.png')
    this_plot.savefig(save_path,facecolor=this_plot.get_facecolor(), edgecolor='none')
    
    mpl.pyplot.close("all")
    #my_plotting.plot_hist_dict(this_run['hist_dict'],title_str).show()

#%% Frame
for run in run_list:
    print(run)    
    
    

