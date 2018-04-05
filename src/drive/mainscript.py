#===============================================================================
#--- SETUP Config
#===============================================================================

from config.config import *

#===============================================================================
#--- SETUP Logging
#===============================================================================
import logging.config

import yaml as yaml
from drive import my_utilities
print(ABSOLUTE_LOGGING_PATH)
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
import copy
import json
import numpy as np

#===============================================================================
#--- SETUP external modules
#===============================================================================
import keras as ks
import sklearn as sk
import sklearn.metrics

#===============================================================================
#--- SETUP Custom modules
#===============================================================================
import ExergyUtilities.util_path

from . import filemanager
from . import my_project
from . import my_callbacks
from . import my_models
from . import my_generators
from . import my_plotting
from . import my_utilities
from . import my_testing

#===============================================================================
#--- Directories and files
#===============================================================================
logging.info("Data path: {}".format(DATA_PATH))
logging.info("Project path: {}".format(PROJECT_PATH))

#===============================================================================
#--- MAIN CODE
#===============================================================================
PROJECT_PATH
DATA_PATH
#DATA_PATH = r"/media/alfred/USB STICK"

def train_model_simple(model,train_generator,validation_generator,callbacks=[]):
    logging.info("Started training".format())
    start_time = time.time()
    history = model.fit_generator(
        train_generator,
        #steps_per_epoch = 20000/50, # Batches??
        steps_per_epoch = 10, # Batches??
        #batch_size = 200,
        #epochs=50,
        epochs=2,
        validation_data = validation_generator,
        #validation_steps = 5000/50,
        validation_steps = 5,
        verbose=0,
        callbacks=callbacks
        )
    logging.info("Elapsed training time {}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
    
#    print("Elapsed:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return history


def train_model(model,train_generator,validation_generator,callbacks=[],epochs=50):
    logging.info("Started training".format())
    start_time = time.time()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 20000/50, # Batches??
        #batch_size = 200,
        epochs=epochs,
        validation_data = validation_generator,
        validation_steps = 5000/50,
        verbose=0,
        callbacks=callbacks
        )
    
    logging.info("Elapsed training time {}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
    
    #print("Elapsed:", )

    return history

def add_project_logger(logger,path_proj):
    fh = logging.FileHandler(filename=os.path.join(path_proj, 'log.txt'))
    fh.setLevel('INFO')
    logformat = logging.Formatter("%(asctime)s - %(levelno)-3s - %(module)-20s  %(funcName)-30s: %(message)s")
    fh.setFormatter(logformat)
    
    
    logger.addHandler(fh)
    
    
    #raise


    return logger 




    
def run(dropout, project_name, data_source_name):

    
    path_proj = my_project.start_project(project_name)

    path_run = my_project.get_next_run_dir(path_proj)

    #--- Add the project logger
    add_project_logger(my_logger,path_run)

    logging.info("Project base: {}".format(PROJECT_PATH))
    logging.info("Started project {}".format(path_proj))
    logging.info("Started run {}".format(path_run))
    
    
    my_utilities.print_tensor_devices()

    #--- Get data paths
    data_root = os.path.join(DATA_PATH,data_source_name)
    
    logging.info("Data located at {}".format(data_root))
    data_dict = filemanager.get_data_set(data_root)
    
    #--- Get generators
    batch_size = 50 
    
    train_generator = my_generators.get_train_generator_simple(data_dict['train'],batch_size)
    
    #train_generator = my_generators.get_train_generator_aug(data_dict['train'],batch_size)

    validation_generator = my_generators.get_validation_generator(data_dict['val'],batch_size)
    
    #--- Get model, and save it
    model = my_models.get_model_4xconv_vary_drop(dropout)
    #model = my_models.get_model_3xconv_vary_drop(dropout)
    #model = my_models.get_model_testing_tiny()
    #model = my_models.get_model_2xconv_vary_drop(dropout)

    
    json_path = os.path.join(path_run,r"saved_model_architecture.json")
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    logging.info("Saved model to {}".format(json_path))
    
    #--- Plot the model architecture to an image
    #my_plotting.image_model(path_run,model)
    
    
    #--- Save the weights at each epoch
    weight_filename="weights-epoch{epoch:02d}-{val_acc:.2f}.hdf5"
    weight_path = os.path.join(path_run,weight_filename)
    checkpt_callback = ks.callbacks.ModelCheckpoint(weight_path, 
                                                    monitor='val_acc', 
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    mode='max')
    
    #--- Callbacks
    history = ks.callbacks.History()
    my_callback = my_callbacks.MyCallback()
    history_filename = "history "+ os.path.split(path_run)[1] + ".txt"
    history_file = os.path.join(path_run,history_filename)
    my_log_callback = my_callbacks.MyLoggingCallback(history_file)
    
    callbacks = [checkpt_callback,history,my_callback,my_log_callback]

    #--- Train the model! 
    history = train_model(model,train_generator,validation_generator,callbacks)
    #history = train_model_simple(model,train_generator,validation_generator,callbacks)
    
    #--- Save the history object
    history_dict = copy.copy(history.__dict__)
    del history_dict['model']
    path_history = os.path.join(path_run,r"saved_model_history.json")
    
    with open(path_history, 'w') as fp:
        json_string = json.dump(history_dict,fp)
    
    logging.info("Saved history.__dict__ to {}".format(path_history))

    logging.info("Finished run {}".format(os.path.split(path_run)[-1]))
    
    #--- Testing
    test_dir_root = data_dict['my_test']
    test_df = my_testing.test_model(model,test_dir_root)
    path_testing_result = os.path.join(path_run,r"saved_testing.csv")
    with open(path_testing_result,'w') as f:
        test_df.to_csv(f)
        
    logging.info("Saved testing to {}".format(path_testing_result))
        
    #print(test_df)

    logging.info("Finished testing {}".format(os.path.split(path_run)[-1]))
    
    
    #--- Metrics
    print_metrics(test_df)

    logging.info("Finished with metric, done {}".format(os.path.split(path_run)[-1]))

    

if __name__ == "__main__":
    #dropout = [0, 0.1, 0.25, 0.5, 0.75]
    project_name='catdog5'
    data_source_name = 'cats_dogs_all_test_split'
    
    
    #run(0.5,project_name,data_source_name)
    
    
    #raise
    
    if 1:
        #dropout = np.arange(0,1,0.1)
        #dropout = np.arange(0,1,0.1)
        dropout = [0.3, 0.4, 0.5, 0.6, 0.7]
        for this_drop in dropout:
            ks.backend.clear_session()
            print(this_drop)
            
            run(this_drop,project_name,data_source_name)
            
        #logging.info("Saved history.__dict__ to {}".format(path_history))

