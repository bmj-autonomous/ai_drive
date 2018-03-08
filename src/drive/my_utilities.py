from tensorflow.python.client import device_lib
#import tensorflow as tf
#from tensorflow.python import pywrap_tensorflow
import re
import logging

#warnings.filterwarnings("default")


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
    
