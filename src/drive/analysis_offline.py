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
#     logging.info("ADDED TO PATH: ".format(this_path))

#===============================================================================
#--- SETUP Standard modules
#===============================================================================
import os
import re
import json
from datetime import datetime

#===============================================================================
#--- SETUP external modules
#===============================================================================
import pandas as pd
import numpy as np
from keras import backend as K

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

#--- Layer output

def Conv2D(params):
    #print(params['class_name'])
    #pprint(params['config'])
    kernel_dim = (params['config']['kernel_size'][0])
    filters = (params['config']['filters'])
    return "{}, kernel {}, filters {}".format(params['class_name'],kernel_dim,filters)

def MaxPooling2D(params):
    #print(params['class_name'])    
    #pprint(params['config'])
    pool_size = (params['config']['pool_size'][0])
    
    return "{}, pool {}".format(params['class_name'],pool_size)

def Flatten(params):
    #print(params['class_name'])
    return "{}".format(params['class_name'])

def Dropout(params):
    #print(params['class_name'])
    #pprint(params['config'])
    drp_rate = (params['config']['rate'])
    return "{}, dropout {}".format(params['class_name'],drp_rate)
    
    
    #raise
def Dense(params):
    #print(params['class_name'])
    return "{}".format(params['class_name'])

LAYER_FUNCS = {
        'Conv2D':Conv2D,
        'MaxPooling2D':MaxPooling2D,
        'Flatten':Flatten,
        'Dropout':Dropout,
        'Dense':Dense,
}



#--- Utilities for log files

def create_date(res_dict):
    """Create the datetime object from the date string
    """
    #values = ['2014', '08', '17', '18', '01', '05']

    datevec = [ int(res_dict['year']),
                int(res_dict['month']),
                int(res_dict['day']),
                int(res_dict['hour']),
                int(res_dict['minute']),
                int(res_dict['second']),
              ]
    
    this_dt = datetime(*datevec)
    return this_dt


def parse_log_string(l):
    """Split into datetime, log string text, module string
    """
    
    #log_regex = re.compile(r"(?P<year>\d{2,4})-(?P<month>\d{2})-(?P<day>\d{2}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}[.\d]*).*:\s(?P<logstring>.*)")
    #-- TODO !!!!
    log_regex = re.compile(r"(?P<year>\d{2,4})-(?P<month>\d{2})-(?P<day>\d{2}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}),[\d]*\s-\s(?P<level>\d\d)\s*-\s(?P<module_str>.*).*:\s(?P<logstring>.*)$")
    #log_regex = re.compile(r"(?P<year>\d{2,4})-(?P<month>\d{2})-(?P<day>\d{2}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}),[\d]*\s-\s(?P<level>\d\d)\s*-\s*(?P<modules>.*)\s:\s(?P<logstring>.*)$")
    res = log_regex.match(l)
    res_dict = res.groupdict()
    
    logstr = res_dict.pop('logstring', None)
    module_str = res_dict.pop('module_str', None)
        #print(res_dict)
    this_dt = create_date(res_dict)    
    
    return({'dt':this_dt,'logstr':logstr,'modstr':module_str})


def get_log_file(this_run_path):
    log_file = [os.path.join(this_run_path,f) for f in os.listdir(this_run_path) 
             if re.match('^log.txt$',f)][0]
    assert os.path.exists(log_file)
    
    with open(log_file) as f:
        lines = f.readlines()
    
    start_time = parse_log_string(lines[0])['dt']
    #reg
    end_time = parse_log_string(lines[-1])['dt']
    elapsed = end_time-start_time
    elapsed_str = None
    
    finish_found = None
    generator = None
    #for l in lines[-100:-1]:
    for l in lines:        
        l = l.strip()
        
        if re.match('\[\d.*\]\]',l): # Workaround for the confusion matrix
            continue
        
        #Elapsed training
        
        res_dict = parse_log_string(l)
        if re.match('Elapsed training time',res_dict['logstr']):
            elapsed_str = res_dict['logstr']
        # Get the image generator
        if not generator: 
            gen_regex = re.compile(r"get_train_generator")
            generator_line = gen_regex.search(res_dict['modstr'])
            #generator_line = gen_regex.match(res_dict['logstr'])
            if generator_line:
                #print(generator_line.group())
                #print()
                generator = res_dict['modstr']
                #raise
            #my_generators         
        
        # Check if finished
        finished_regex = re.compile(r"Finished training")
        finished_line = finished_regex.match(res_dict['logstr'])
        if finished_line:
            
            finish_found=True
    
    return({'finished':finish_found,
            'start':start_time,
            'end':end_time,
            'elapsed':elapsed,
            'elapsed_str':elapsed_str,
            'generator':generator})
    #assert(finish_found)




#--- Model object

def get_architecture_path(this_run_path):
    arch_file = [os.path.join(this_run_path,f) for f in os.listdir(this_run_path) 
             if re.match('saved_model_architecture.json$',f)]
    if arch_file: arch_file = arch_file.pop() 
    else: arch_file = None
    #print(arch_file)
    
    logging.debug("Found architecture file at {}".format(arch_file))
    
    return(arch_file)

def read_model_json(this_run_path):
    """Read the saved model as a json string
    """
    path_arch = get_architecture_path(this_run_path)
    with open(path_arch,'r') as arch_file:
        arch_dict = json.load(arch_file)
    
    logging.debug("Model json string loaded".format())
    
    return arch_dict

def load_model(this_run_path):
    """Load the model json and instantiate it in Keras
    """
    path_arch = get_architecture_path(this_run_path)
    from keras.models import model_from_json
    
    with open(path_arch,'r') as arch_file:
        loaded_model_json = arch_file.read()
        model = model_from_json(loaded_model_json)
    
    logging.debug("Model instantiated {}".format(model))
    
    return model


def count_params(model):
    """Uses backend 'K' count_params
    Send model object
    Returns dict for Total/Trainable/Non-trainable
    """
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    
    logging.debug("Total {}, Trainable {}, Non-Trainable".format(trainable_count + non_trainable_count,
                                                                 trainable_count,
                                                                 non_trainable_count))
    
    return {'Total':trainable_count + non_trainable_count,
            'Trainable':trainable_count,
            'Non-trainable':non_trainable_count}

    #print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    #print('Trainable params: {:,}'.format(trainable_count))
    #print('Non-trainable params: {:,}'.format(non_trainable_count))
    #```total_params = (filter_height * filter_width * input_image_channels + 1) * number_of_filters```
    

#--- Weights

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

    logging.info("Found {} weights files, total {:.0f} MB = {:.1f} MB per file".format(
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

#--- History log file

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
    
    logging.debug("History file loaded {}".format(hist_dict))
    
    return hist_dict    

#--- OTHERS

def get_solutions_csv(path_solutions):
    df_solutions = pd.read_csv(path_solutions)
    df_solutions.head()
    #solutions.[]
    cutoff=0.5
    df_solutions['labelTF'] = df_solutions['label'].map(lambda x: True if x >= 0.5 else False)
    df_solutions.set_index('id',inplace=True)
    df_solutions.sort_index(inplace=True)
    #df_solutions.head()
    logging.info("Loaded solutions from {}, {} rows".format(path_solutions,len(df_solutions)))
    return df_solutions


def run():
    #xrg_path.
    #os.list(PROJECT_PATH)
    this_project_path = r"/media/batman/USB STICK"
    project_name = r'catdog1'
    root_path = os.path.join(this_project_path,project_name)
    run_folders = [dir for dir in os.listdir(root_path) if re.match('run',dir)]
    for rfolder in run_folders:
        print(rfolder)
        logging.info("In {}".format(rfolder))
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
