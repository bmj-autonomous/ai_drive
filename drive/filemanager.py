#===============================================================================
#--- SETUP Config
#===============================================================================
#import sys 

#print('\n'.join(sorted(sys.path))) 

#/home/batman/git/py_ExergyUtilities/ExergyUtilities
import ExergyUtilities as xrg
import ExergyUtilities.util_path

#import ExergyUtilites.util_excel
#import ExergyUtilites as x

from config.config import *
import unittest

#===============================================================================
#--- SETUP Logging
#===============================================================================
import logging.config
print(ABSOLUTE_LOGGING_PATH)
import yaml as yaml
log_config = yaml.load(open(ABSOLUTE_LOGGING_PATH, 'r'))
logging.config.dictConfig(log_config)

myLogger = logging.getLogger()
myLogger.setLevel("DEBUG")

#===============================================================================
#--- SETUP standard modules
#===============================================================================
import os
from pprint  import pprint
#===============================================================================
#--- SETUP external modules
#===============================================================================

#===============================================================================
#--- SETUP Custom modules
#===============================================================================

#===============================================================================
#--- Directories and files
#===============================================================================
logging.debug("Data path: {}".format(DATA_PATH))
logging.debug("Project path: {}".format(PROJECT_PATH))

#===============================================================================
#--- MAIN CODE
#===============================================================================

def get_data_set(this_path):
    path_dict = dict()
    
    # Get the three paths
#     path_train  = os.path.join(this_path,'train')
#     path_test   = os.path.join(this_path,'test')
#     path_val    = os.path.join(this_path,'validation')
#     assert(os.path.exists(path_train)), "DNE"
#     assert(os.path.exists(path_test)), "DNE"
#     assert(os.path.exists(path_val)), "DNE"
    
# Get the three paths
    subdirs = [thissubdir for thissubdir in os.listdir(this_path) if  
             os.path.isdir(os.path.join(this_path,thissubdir))]
    assert 'train' in subdirs
    assert 'test' in subdirs
    assert 'validation' in subdirs
    
    for subdir in subdirs:
        subpath = os.join()
     
    
    
def get_path_dict(this_path):
    data_dict = dict()
    
    for (path, dirs, files) in os.walk(this_path):
        files = [file for file in files]
        
        if len(files) > 0:
            total_split = xrg.util_path.my_splitpath(path)
            variablename = "_".join(total_split[-2:])
            data_dict[variablename] = dict()
            data_dict[variablename]['path'] = path
            data_dict[variablename]['file_count'] = len(files)

    logging.debug("File and path count dictionary returned from {}".format(this_path))
    return data_dict

def run():
    data_root = os.path.join(DATA_PATH,'cats_dogs_small')
    logging.debug("Data located at {}".format(data_root))
    #xrg.util_path.my_splitpath(DATA_PATH)
    #data_root_split = ExergyUtilities.util_path.my_splitpath(data_root)
    #print(data_root_split)
    
    #path_dict=get_path_dict(data_root)
    path_dict=get_data_set(data_root)
    
#     print("FILE OVERVIEW")
#     for k in path_dict:
#         
#         print(k, path_dict[k])
#         




if __name__ == "__main__":
    run()
