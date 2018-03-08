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
#from drive import my_generators
log_config = yaml.load(open(ABSOLUTE_LOGGING_PATH, 'r'))
logging.config.dictConfig(log_config)

my_logger = logging.getLogger()
my_logger.setLevel("DEBUG")

#===============================================================================
#--- SETUP standard modules
#===============================================================================
import os
from pprint import pprint
import threading
import time
import warnings
import re

#===============================================================================
#--- SETUP external modules
#===============================================================================
import numpy as np
import pandas as pd

# Disable TensorFlow warnings
warnings.filterwarnings("ignore", 'FutureWarning')
warnings.filterwarnings("ignore", 'DeprecationWarning')

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.python.client import device_lib
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import keras as ks
#from keras.preprocessing import image

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg 


#===============================================================================
#--- SETUP Custom modules
#===============================================================================
#import ExergyUtilities as xrg
import ExergyUtilities.util_path


# import filemanager
# import project
# import my_callbacks
# import my_models
# import my_generators
# import my_plotting


#===============================================================================
#--- Directories and files
#===============================================================================
logging.debug("Data path: {}".format(DATA_PATH))
logging.debug("Project path: {}".format(PROJECT_PATH))

#===============================================================================
#--- MAIN CODE
#===============================================================================
warnings.filterwarnings("default")


#from . import my_generatorsXX
#from . import foo
#print(foo)
print(os.sys.path)
#print(drive.my_models)
#print(drive.my_generators)
print(drive.my_generatorsXX)

#print(drive.foo)
#print(drive.my_generators.get_train_generator_simple)
#print(drive.foo)
#print(drive.my_project2)
raise



def train_model(model,train_generator,validation_generator,callbacks=[]):
    logging.debug("Started training".format())
    start_time = time.time()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 2, # Batches
        epochs=10,
        validation_data = validation_generator,
        validation_steps = 2,
        verbose=0,
        callbacks=callbacks
        )
    
    print("Elapsed:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def print_tensor_devices():
    devices_listing = device_lib.list_local_devices()
    
    devices = list()
    for dev in devices_listing:
        this_dev = str(dev)
        
        dev_dict = dict()
        for item in this_dev.split('\n'):
            if re.search(r':\s',item):
                pair = re.split(r':\s',item)
                dev_dict[pair[0]] = pair[1]
                #print(pair)
                
        devices.append(dev_dict)
    
    for i,dev in enumerate(devices):
        logging.debug("Device {}, {}, type {}, memory {}".format(i,
            dev['name'],
            dev['device_type'],
            dev['memory_limit'],            ))
    



    
def run():
    project_name='catdog1'
    
    
    path_proj = drive.my_project.start_project(project_name)
    
    # Add the project logger
    drive.my_project.add_project_logger(my_logger,path_proj)
    logging.debug("Project base: {}".format(PROJECT_PATH))
    logging.debug("Started project {}".format(path_proj))
    
    
    path_run = drive.my_project.get_next_run_dir(path_proj)
    
    data_root = os.path.join(DATA_PATH,'cats_dogs_all')
    
    logging.debug("Data located at {}".format(data_root))
    data_dict = drive.filemanager.get_data_set(data_root)
    
    print_tensor_devices()
    
    train_generator = my_generators.get_train_generator_simple(data_dict['train'])
    validation_generator = my_generators.get_validation_generator(data_dict['val'])
    
    model = my_models.get_model()

    # Save the weights at each epoch
    weight_filename="weights-epoch{epoch:02d}-{val_acc:.2f}.hdf5"
    weight_path = os.path.join(path_run,weight_filename)
    checkpt_callback = ks.callbacks.ModelCheckpoint(weight_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    history = ks.callbacks.History()
#     print(history)
#     print(type(history))
#     for i in dir(history):
#         print(i)
    # This is my callback
    my_callback = my_callbacks.MyCallback()
    history_filename = "history "+ os.path.split(path_run)[1] + ".txt"
    #print(history_filename)
    #raise
    history_file = os.path.join(path_run,history_filename)
    my_log_callback = my_callbacks.MyLoggingCallback(history_file)
    
    
    callbacks = [checkpt_callback,history,my_callback,my_log_callback]
        
    train_model(model,train_generator,validation_generator,callbacks)
    
    
if __name__ == "__main__":
    
    run()
