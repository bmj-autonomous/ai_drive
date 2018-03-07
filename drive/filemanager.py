#===============================================================================
#--- SETUP Config
#===============================================================================

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
import glob
import random
import math
import shutil
from timeit import default_timer as timer

#===============================================================================
#--- SETUP external modules
#===============================================================================
import pandas as pd
from tabulate import tabulate


#===============================================================================
#--- SETUP Custom modules
#===============================================================================

import ExergyUtilities as xrg
import ExergyUtilities.util_path

#===============================================================================
#--- Directories and files
#===============================================================================

#===============================================================================
#--- MAIN CODE
#===============================================================================

def get_data_set(base_path):
    """Organize data set files into a dictionary
    base
    classes
    subdirs
    filecounts
        test 
            cats
            dogs
        train
        val
    folders
        test 
            cats
            dogs
        train
        val    
    """
    from collections import defaultdict
    dict_data_path = defaultdict(dict)
    
    dict_data_path['base'] = base_path

    subdirs = [thissubdir for thissubdir in os.listdir(base_path) if  
                    os.path.isdir(os.path.join(base_path,thissubdir))]
    
    dsets = ['train', 'test', 'val']

    assert all([d in subdirs for d in dsets]), f"Missing a dataset {subdirs}"
    dict_data_path['subdirs'] = subdirs


    logging.debug("Found [train, test, val] in {}".format(base_path))
    
    # Get the classes
    classdirsall = list()
    for subdir in subdirs:
        subpath = os.path.join(base_path,subdir)
        
        classdirs = [thisclass for thisclass in os.listdir(subpath) if  
                     os.path.isdir(os.path.join(subpath,thisclass))]
        
        classdirsall.append(classdirs)

    compare = [x >= y for i,x in enumerate(classdirsall) for j,y in enumerate(classdirsall) if i > j]
    
    assert all(compare), "Mismatch"
    classes = classdirsall.pop()
    logging.debug("Found {} classes: {}".format(len(classes),classes))
    dict_data_path['classes'] = classes
    
    dict_data_path['folders'] = dict()
    dict_data_path['filecounts'] = dict()
    
    
    for subdir in dict_data_path['subdirs']:
        dict_data_path['folders'][subdir] = dict()
        dict_data_path['filecounts'][subdir] = dict()
        dict_data_path[subdir] = os.path.join(dict_data_path['base'],subdir)
        
        for this_class in dict_data_path['classes']:
            #print(this_class)
            this_path = os.path.join(dict_data_path['base'],subdir,this_class)
            dict_data_path['folders'][subdir][this_class] = this_path
            
            file_cnt = len(glob.glob(this_path + '/*.jpg'))
            dict_data_path['filecounts'][subdir][this_class] = file_cnt
    
    #pprint(dict_data_path['filecounts'])
    df = pd.DataFrame.from_dict(dict_data_path['filecounts'])
    logging.debug("Organized dataset paths at {}".format(base_path))
    
    print(tabulate(df, headers='keys', tablefmt='psql'))


    return(dict_data_path)

def run():
    data_root = os.path.join(DATA_PATH,'cats_dogs_all')
    
    logging.debug("Data located at {}".format(data_root))
    
    path_dict=get_data_set(data_root)


if __name__ == "__main__":
    logging.debug("Data path: {}".format(DATA_PATH))
    logging.debug("Project path: {}".format(PROJECT_PATH))
    run()
