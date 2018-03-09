import numpy as np
import os
import matplotlib.pyplot as plt
import logging 
import matplotlib.image as mpimg 
import keras as ks
#import pydot
#import graphviz

def image_model(path_save,model):
    this_path = os.path.join(path_save,'modelLRnamed.png')
    ks.utils.plot_model(model, to_file=this_path, show_shapes=True, show_layer_names=True, rankdir='LR')
    this_path = os.path.join(path_save,'modelTBnamed.png')
    ks.utils.plot_model(model, to_file=this_path, show_shapes=True, show_layer_names=True, rankdir='TB')

    this_path = os.path.join(path_save,'modelLR.png')
    ks.utils.plot_model(model, to_file=this_path, show_shapes=True, show_layer_names=False, rankdir='LR')
    this_path = os.path.join(path_save,'modelTB.png')
    ks.utils.plot_model(model, to_file=this_path, show_shapes=True, show_layer_names=False, rankdir='TB')
    
    logging.debug("Saved model images to {}".format(path_save))

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


if __name__ == '__main__':
    pass