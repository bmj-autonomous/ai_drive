#===============================================================================
#--- SETUP Config
#===============================================================================
from config.config import *
#import unittest

#===============================================================================
#--- SETUP Logging
#===============================================================================
import logging
#import logging.config

#import yaml as yaml
#log_config = yaml.load(open(ABSOLUTE_LOGGING_PATH, 'r'))
#logging.config.dictConfig(log_config)

#myLogger = logging.getLogger()
#myLogger.setLevel("DEBUG")

#===============================================================================
#--- SETUP Add parent module
#===============================================================================
# # Add parent to path
# if __name__ == '__main__' and __package__ is None:
#     this_path = path.dirname(path.dirname(path.abspath(__file__)))
#     sys.path.append(this_path)
#     logging.debug("ADDED TO PATH: ".format(this_path))

#===============================================================================
#--- SETUP Standard modules
#===============================================================================
import os
from os import sys, path
import re

#===============================================================================
#--- SETUP external modules
#===============================================================================

#===============================================================================
#--- SETUP Custom modules
#===============================================================================
#import ExergyUtilities.util_path
from ExergyUtilities import util_path as util_path


#===============================================================================
#--- Directories and files
#===============================================================================
#curr_dir = path.dirname(path.abspath(__file__))
#DIR_SAMPLE_IDF = path.abspath(curr_dir + "\..\.." + "\SampleIDFs")
#print(DIR_SAMPLE_IDF)

#===============================================================================
#--- MAIN CODE
#===============================================================================
def add_project_logger(logger,path_proj):
    fh = logging.FileHandler(filename=path.join(path_proj, 'log.txt'))
    fh.setLevel('DEBUG')
    logformat = logging.Formatter("%(asctime)s - %(levelno)-3s - %(module)-20s  %(funcName)-30s: %(message)s")
    fh.setFormatter(logformat)
    logger.addHandler(fh)

def get_next_run_dir(path_proj):
    # List directories in this project
    subdirs = [thissubdir for thissubdir in os.listdir(path_proj) if  
                path.isdir(path.join(path_proj,thissubdir))]
    
    # Get the next iteration of the run
    last_count = -1
    for dir in subdirs:
        if re.match('run\d\d',dir):
            digit_string = re.search('\d\d\d',dir).group(0)
            this_count = int(digit_string)
            if this_count>=last_count: 
                last_count = this_count
    next_count = last_count + 1
    next_digit_string = str(next_count).zfill(3)
    
    # Make the directory
    dir_next_run = path.join(path_proj,'run'+next_digit_string)
    os.mkdir(dir_next_run)
    logging.debug("Made next run directory at {}".format(dir_next_run))
    return dir_next_run
    
def run(path_proj):
    pass

def start_project(project_name):
    # Check the project name and directory
    path_proj=path.join(PROJECT_PATH,project_name)
    assert path.exists(path_proj)
    

    return path_proj
    
if __name__ == "__main__":
    project_name='catdog1'
    
    path_proj = start_project(project_name)
    
    path_run = get_next_run_dir(path_proj)
    
    #run(path_proj)
