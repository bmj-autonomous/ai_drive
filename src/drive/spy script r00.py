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
import drive.my_testing as my_testing

print(analysis)

#%% Utility

def dflatten(current, key, result):
    if isinstance(current, dict):
        for k in current:
            new_key = "{0}.{1}".format(key, k) if len(key) > 0 else k
            dflatten(current[k], new_key, result)
    else:
        result[key] = current
    return result


#%% Paths 

# Project path
this_project_path = r"/media/batman/SABRINI/PROJECT"
#project_name = r'catdog6'
#path_root_project = os.path.join(this_project_path,project_name)
assert os.path.exists(this_project_path)

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
#run_folders = [dir for dir in os.listdir(path_root_project) if re.match('run',dir)]
#run_folders.sort()



#%% Next
run_list = list()

project_names = ['catdog5','catdog6','catdog7','catdog8']

for project_name in project_names:
    print('******************', project_name)
    path_root_project = os.path.join(this_project_path,project_name)
    assert os.path.exists(path_root_project), path_root_project
    run_folders = [dir for dir in os.listdir(path_root_project) if re.match('run',dir)]
    run_folders.sort()
        
    # Loop run folders
    for rfolder in run_folders:
        
        summary=dict()
        
        summary['project name'] = project_name
        
        this_run_path = os.path.join(path_root_project,rfolder)
        logging.debug('**** RUN {} ****'.format(rfolder))
        
        summary['run'] = rfolder
        
        runnum = re.findall(r'\d+', summary['run'])[0]
        summary['runi'] = int(runnum)
        
        ###### Log file ######
        log = analysis.get_log_file(this_run_path)
        
        #print('start;',log['start'])
        summary['log'] = log
        summary['start'] = log['start'].__str__()
        #print('elapsed; {:.1f}'.format(log['elapsed'].seconds/60))
        summary['elapsed'] = log['elapsed'].seconds/60
        #print('generator;',log['generator'])
        summary['generator'] = log['generator']
        
        #del log
        
        ###### Weights ######
        wts = analysis.get_weights(this_run_path)
        best_wt = wts[-1] # BEST weight (last weight)
        del wts # Save space
        
        ###### Architecture ######
        model = analysis.load_model(this_run_path)
        summary['arch_dict'] = analysis.read_model_json(this_run_path)
        #model.summary()
        parameter_counts = analysis.count_params(model)
        summary['param_counts'] = parameter_counts
        #print(analysis.count_params(model))
        summary['param_cnt']  = summary['param_counts']['Total']
        
        ##### Initialize model, reload weights, best #####
        model.load_weights(best_wt['path'])
        logging.debug("Loaded weights into model")
        
    
        ###### Loop layers ######
        for i,layer in enumerate(summary['arch_dict']['config']):
            if layer['class_name'] == 'Dropout':
                #print(layer['config']['rate'])
                summary['Dropout'] = layer['config']['rate']
                
        layer_summary = defaultdict(int)
        for layer in summary['arch_dict']['config']:
            layer_summary[layer['class_name']] += 1
        summary['layers'] = layer_summary
        
        ###### History ######
        path_hist = analysis.get_history(this_run_path)    
        with open(path_hist) as hist_file:
            hist_dict = json.load(hist_file)
        summary['hist_dict'] = hist_dict
        
        ###### Re-Testing ######
        if 0:
            #test_dir_root = r"/home/batman/Dropbox/DATA/cats_dogs_all_test_split/my_test_tiny"
            test_dir_root = r"/home/batman/Dropbox/DATA/cats_dogs_all_test_split/my_test"
            test_df = my_testing.test_model(model,test_dir_root)
            path_testing_result = os.path.join(this_run_path,r"saved_testing.csv")
            with open(path_testing_result,'w') as f:
                test_df.to_csv(f)
            
            my_testing.print_metrics(test_df)
    
        ###### Load Testing ######
        
        path_testing_result = os.path.join(this_run_path,r"saved_testing.csv")
        assert os.path.exists(path_testing_result)
        with open(path_testing_result,'r') as f:
            test_df = pd.read_csv(f)
        
        summary['metrics'] = my_testing.get_metrics(test_df)
        
        #break
        ###### Summary string ######
        assert summary['layers']['Conv2D'] == summary['layers']['MaxPooling2D']
        summary['num_convpool'] = summary['layers']['Conv2D']
        summary['title_str'] = '{} - {}xConvPool, {} dropout, {:0.1f}M parameters'.format(
                summary['runi'],
                summary['num_convpool'],
                summary['Dropout'],
                summary['param_cnt']/1000000)
        
        
        ###### Done ######
        run_list.append(summary)
        
        del model
    
        #break


#%% Plot
    
#import matplotlib as mpl
#mpl.use('Agg')
#this_run = run_list[12]
if 0:
    for this_run in run_list:
    
        #title_str.append()
        
        this_plot = my_plotting.plot_hist_dict(this_run['hist_dict'],summary['title_str'])
        save_path = os.path.join(path_root_project,summary['title_str']+'.png')
        this_plot.savefig(save_path,facecolor=this_plot.get_facecolor(), edgecolor='none')
        
        mpl.pyplot.close("all")
        #my_plotting.plot_hist_dict(this_run['hist_dict'],title_str).show()

#%% Frame
d = list()

#new = 

for run in run_list:
    print('Appended',run['project name'],run['run'])
    #run.pop('hist_dict',None)
    #run.pop('param_counts',None)
    #run.pop('layers',None)
    d.append(dflatten(run,'', {}))
    
    
df = pd.DataFrame(d)

#%% Export
#from pandas import ExcelWriter

writer = pd.ExcelWriter('/media/batman/USB STICK/PythonExport.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()


#%% Analysis


df2 = df.copy() # Create a copy
# Reindex
df2.set_index(['num_convpool', 'Dropout'], inplace=True)

# Discard other cols
keepcols = list()
keepcols+=[col for col in df2.columns if re.match('metrics',col)]
keepcols+=[col for col in df2.columns if re.match('hist_dict',col)]
keepcols+=['param_counts.Total']


df2.index
df2 = df2.filter(keepcols)

writer = pd.ExcelWriter('/media/batman/USB STICK/PythonExport2.xlsx')
df2.to_excel(writer,'Sheet1')
writer.save()

aa = df2.query('num_convpool==4')


col_logloss = df.groupby(['num_convpool', 'Dropout'])['metrics.log_loss'].mean()
col_logloss.name = 'mean logloss'

col_f1 = df.groupby(['num_convpool', 'Dropout'])['metrics.f1_score'].mean()
col_f1.name = 'mean F1'


a1 = df.groupby(['num_convpool', 'Dropout'])['metrics.accuracy_score'].mean()
a1.name = 'mean accuracy'
a2 = df.groupby(['num_convpool', 'Dropout'])['metrics.accuracy_score'].std()
a2.name = 'std accuracy'
a3 = df.groupby(['num_convpool', 'Dropout'])['metrics.accuracy_score'].count()
a3.name = 'count accuracy'

frames = [a1, a2, a3,col_logloss,col_f1]
df_result = pd.concat(frames, axis=1)


a = df2.loc[3]


























