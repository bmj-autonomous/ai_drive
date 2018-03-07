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
from nltk.sem.logic import printtype
from gevent.libev.corecffi import callback
log_config = yaml.load(open(ABSOLUTE_LOGGING_PATH, 'r'))
logging.config.dictConfig(log_config)

myLogger = logging.getLogger()
myLogger.setLevel("DEBUG")

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
from keras.preprocessing import image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 


#===============================================================================
#--- SETUP Custom modules
#===============================================================================
#import ExergyUtilities as xrg
import ExergyUtilities.util_path

import filemanager

#===============================================================================
#--- Directories and files
#===============================================================================
logging.debug("Data path: {}".format(DATA_PATH))
logging.debug("Project path: {}".format(PROJECT_PATH))

#===============================================================================
#--- MAIN CODE
#===============================================================================
warnings.filterwarnings("default")


class LossHistory(ks.callbacks.Callback):
    def on_train_begin(self, logs={}):
        logging.debug("Started training {}".format(self.model))
        self.losses = []
        return 
        
    def on_train_end(self, logs={}):
        logging.debug("Finished training {}".format(self.model))
        return
 

    def on_epoch_begin(self, epoch, logs={}):
        logging.debug("Epoch {}".format(epoch))
        
        return
 
    def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        logging.debug("\tBatch {} {}".format(batch,logs))

        
    def on_epoch_end(self, epoch, logs={}):
        logging.debug("*".format(epoch))
        

class My_Callback(ks.callbacks.Callback):
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        #raise
        return
 
    def on_batch_begin(self, batch, logs={}):
        
        return
 
    def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        return


def plot_sample(data_dict):
    display_images = list()
    
    # Get cats
    this_path = data_dict['folders']['train']['cats']
    these_files = [os.path.join(this_path,f) for f in os.listdir(this_path) if os.path.isfile(os.path.join(this_path, f))]
    np.random.shuffle(these_files)
    display_images += these_files[0:4]
    
    # Get dogs
    this_path = data_dict['folders']['train']['dogs']
    these_files = [os.path.join(this_path,f) for f in os.listdir(this_path) if os.path.isfile(os.path.join(this_path, f))]
    np.random.shuffle(these_files)
    display_images += these_files[0:4]    
    color = (17/255,17/255,17/255)
    fig=plt.figure(figsize=(20, 10),facecolor=color)
    columns = 4
    rows = 2
    for i in range(1, columns*rows +1):
        this_img_path = display_images[i - 1]
        fname = os.path.split(this_img_path)[-1]
        name, number = fname.split(".")[:2]
        #img=mpimg.imread(this_img_path)
        img=plt.imread(this_img_path)
        this_ax = fig.add_subplot(rows, columns, i)
        
        this_ax.set_title("{} {} {}".format(name, number, img.shape,))
        #plt.imshow(img)
        #img = Image.open('IMG_0007.jpg')
        
        this_ax.imshow(img)
        #import Image
        
        
        #img.split()
        plt.axis("off")
    logging.debug("Displaying images".format())
    #plt.show()

def get_model():
    
    model = ks.models.Sequential()

    model.add(ks.layers.Conv2D(32, (3,3), activation = "relu", input_shape=(150,150,3)))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Conv2D(64, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    #model.add(ks.layers.Conv2D(128, (3,3), activation = "relu"))
    #model.add(ks.layers.MaxPooling2D(2,2))
    
    #model.add(ks.layers.Conv2D(128, (3,3), activation = "relu"))
    #model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Flatten()) # This is just a reshape!
    
    model.add(ks.layers.Dropout(0.5))
    
    model.add(ks.layers.Dense(512,activation="relu"))
    model.add(ks.layers.Dense(1,activation="sigmoid"))

    #model.summary()
    model.compile(   
    optimizer = ks.optimizers.RMSprop(lr=0.0001),
    loss= ks.losses.binary_crossentropy,
    metrics= ["accuracy"],
    )
    
    
    return model



def train_model(model,train_generator,validation_generator,callbacks=[]):
    logging.debug("Started training".format())
    start_time = time.time()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 3,
        epochs=2,
        validation_data = validation_generator,
        validation_steps = 2,
        verbose=0,
        callbacks=callbacks
        )
    
    print("Elapsed:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def get_train_generator_aug():
    # Training generator - Augmentation
    train_datagen = ks.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                  rotation_range = 40,
                                                  width_shift_range = 0.2,
                                                  height_shift_range = 0.2,
                                                  shear_range = 0.2,
                                                  zoom_range= 0.2,
                                                  verbose=0,
                                                  horizontal_flip=True)

def get_validation_generator(directory):
    # Validation images
    validation_datagen = ks.preprocessing.image.ImageDataGenerator(rescale=1/255)
    validation_generator = validation_datagen.flow_from_directory(
        directory,
        target_size = (150,150),
        batch_size = 20,
        class_mode = "binary",
    )
    
    logging.debug("Validation: {} files over {} classes, resized to {}".format(
        len(validation_generator.filenames),
        validation_generator.num_classes,
        validation_generator.target_size,
        ))
    
    return validation_generator

def get_train_generator_simple(directory):
    train_datagen = ks.preprocessing.image.ImageDataGenerator(rescale=1/255)
    # Training images
    
    train_generator = train_datagen.flow_from_directory(
        directory,
        target_size = (150,150),
        batch_size = 20,
        class_mode = "binary",
    );
    
    logging.debug("Training: {} files over {} classes, resized to {}".format(
        len(train_generator.filenames),
        train_generator.num_classes,
        train_generator.target_size,
        ))

    return train_generator

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
        #print(devices)
    
    
def run():
    data_root = os.path.join(DATA_PATH,'cats_dogs_all')
    
    logging.debug("Data located at {}".format(data_root))
    data_dict = filemanager.get_data_set(data_root)
    
    print_tensor_devices()
    
    train_generator = get_train_generator_simple(data_dict['train'])
    validation_generator = get_validation_generator(data_dict['val'])
    
    model = get_model()

    this_callback = My_Callback()
    this_callback = LossHistory()
    callbacks = [this_callback]
        
    train_model(model,train_generator,validation_generator,callbacks)
    
    
if __name__ == "__main__":

    
    run()
