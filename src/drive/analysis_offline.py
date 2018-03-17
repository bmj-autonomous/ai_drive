#===============================================================================
#--- SETUP Config
#===============================================================================
from config.config import *
#import unittest

#===============================================================================
#--- SETUP Logging
#===============================================================================
import logging.config

import yaml as yaml
log_config = yaml.load(open(ABSOLUTE_LOGGING_PATH, 'r'))
logging.config.dictConfig(log_config)

myLogger = logging.getLogger()
myLogger.setLevel("DEBUG")

#===============================================================================
#--- SETUP Add parent module
#===============================================================================
# from os import sys, path
# # Add parent to path
# if __name__ == '__main__' and __package__ is None:
#     this_path = path.dirname(path.dirname(path.abspath(__file__)))
#     sys.path.append(this_path)
#     logging.debug("ADDED TO PATH: ".format(this_path))

#===============================================================================
#--- SETUP Standard modules
#===============================================================================
import os
import re
import json

#===============================================================================
#--- SETUP external modules
#===============================================================================

#===============================================================================
#--- SETUP Custom modules
#===============================================================================
import ExergyUtilities.util_path as xrg_path

#===============================================================================
#--- Directories and files
#===============================================================================
#curr_dir = path.dirname(path.abspath(__file__))
#DIR_SAMPLE_IDF = path.abspath(curr_dir + "\..\.." + "\SampleIDFs")
#print(DIR_SAMPLE_IDF)

#===============================================================================
#--- MAIN CODE
#===============================================================================

PROJECT_PATH
DATA_PATH

def get_weights(this_run_path):
    """Return the weights hdf5 files from a directory
    Creates a sorted dictionary (first to last)
    epoch
    fname
    path
    size
    """
    # File name
    wts_file_name = [f for f in os.listdir(this_run_path)
                     if re.match('weights-',f)]
    
    # Total number of files
    length = len([i for i in wts_file_name])
    
    # File path
    wts_file_path = [os.path.join(this_run_path,f) for f in os.listdir(this_run_path) 
                     if re.match('weights-',f)]
    
    # File sizes
    sizes = [os.path.getsize((os.path.join(this_run_path,f)))/1024/1024 
            for f in os.listdir(this_run_path) 
            if re.match('weights-',f)]

    total_size = sum(sizes)
    if not total_size == 0:
        avg_size = sum(sizes)/length
    else:
        avg_size = 0
    
    # Epoch numbers
    epochs = [re.findall(r'epoch\d+', f)[0] for f in wts_file_name]
    epoch_num = [int(re.findall(r'\d+', f)[0]) for f in epochs]
    #print(epoch_num)

    logging.debug("Found {} weights files, total {:.0f} MB = {:.1f} MB per file".format(
        length,total_size,avg_size
        )
    )

    weights_files = list(zip(wts_file_name,wts_file_path,sizes,epoch_num))
    
    # Sort
    weights_files.sort(key=lambda tup: tup[3])
    
    # Convert to dict
    wt_dicts = list()
    wt_dicts = [{'epoch':i[3],'fname':i[0], 'path':i[1],'size':i[2]} for i in weights_files] 

    return wt_dicts


def get_architecture_path(this_run_path):
    arch_file = [os.path.join(this_run_path,f) for f in os.listdir(this_run_path) 
             if re.match('saved_model_architecture.json$',f)]
    if arch_file: arch_file = arch_file.pop() 
    else: arch_file = None
    #print(arch_file)
    return(arch_file)


def read_model_json(this_run_path):
    """Read the saved model as a json string
    """
    path_arch = get_architecture_path(this_run_path)
    with open(path_arch,'r') as arch_file:
        arch_dict = json.load(arch_file)
    
    return arch_dict

def load_model(this_run_path):
    """Load the model and instantiate it in Keras
    """
    path_arch = get_architecture_path(this_run_path)
    from keras.models import model_from_json
    
    with open(path_arch,'r') as arch_file:
        loaded_model_json = arch_file.read()
        model = model_from_json(loaded_model_json)
        
    return model
        

def get_history(this_run_path):
    hist_file_simple = [os.path.join(this_run_path,f) for f in os.listdir(this_run_path) 
                 if re.match('history run\d\d\d.txt$',f)]
    if hist_file_simple: hist_file_simple = hist_file_simple.pop() 
    else: hist_file_simple = None
    print(hist_file_simple)
    
    hist_dict = [os.path.join(this_run_path,f) for f in os.listdir(this_run_path)
                 if re.match('saved_model_history.json$',f)]
    if hist_dict: hist_dict = hist_dict.pop() 
    else: hist_dict = None
            
    return hist_dict    

def run():
    #xrg_path.
    #os.list(PROJECT_PATH)
    this_project_path = r"/media/batman/USB STICK"
    project_name = r'catdog1'
    root_path = os.path.join(this_project_path,project_name)
    run_folders = [dir for dir in os.listdir(root_path) if re.match('run',dir)]
    for rfolder in run_folders:
        print(rfolder)
        logging.debug("In {}".format(rfolder))
        this_run_path = os.path.join(root_path,rfolder)
        get_weights(this_run_path)

        #raise

        #raise
            
        
        # Get the log file
        log_file = [os.path.join(this_run_path,f) for f in os.listdir(this_run_path) 
                    if re.match('log.txt$',f)]
        assert len(log_file)<=1
        if log_file: log_file = log_file.pop() 
        else: log_file = None        
        print(log_file)
        
        # Get the architecture 

        # Get the history

        #hist_dict = 
        
        #json.load(fp)
        
        #print(fpaths)



if __name__ == "__main__":
    run()
