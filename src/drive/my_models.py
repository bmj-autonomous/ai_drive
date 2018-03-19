'''
Created on Mar 8, 2018

@author: batman
'''
import keras as ks
import logging
def get_model():
    dropout = 0.5
    logging.debug("Model with dropout {}".format(dropout))

    model = ks.models.Sequential()

    model.add(ks.layers.Conv2D(32, (3,3), activation = "relu", input_shape=(150,150,3)))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Conv2D(64, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Conv2D(128, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))

    model.add(ks.layers.Conv2D(128, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Flatten()) # This is just a reshape!
    
    model.add(ks.layers.Dropout(dropout))
    
    model.add(ks.layers.Dense(512,activation="relu"))
    model.add(ks.layers.Dense(1,activation="sigmoid"))

    #model.summary()
    model.compile(   
    optimizer = ks.optimizers.RMSprop(lr=0.0001),
    loss= ks.losses.binary_crossentropy,
    metrics= ["accuracy"],
    )
    
    
    return model


def get_model_4xconv_vary_drop(dropout):
    logging.debug("Model with dropout {}".format(dropout))

    model = ks.models.Sequential()

    model.add(ks.layers.Conv2D(32, (3,3), activation = "relu", input_shape=(150,150,3)))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Conv2D(64, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Conv2D(128, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))

    model.add(ks.layers.Conv2D(128, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Flatten()) # This is just a reshape!
    
    model.add(ks.layers.Dropout(dropout))
    
    model.add(ks.layers.Dense(512,activation="relu"))
    model.add(ks.layers.Dense(1,activation="sigmoid"))

    #model.summary()
    model.compile(   
    optimizer = ks.optimizers.RMSprop(lr=0.0001),
    loss= ks.losses.binary_crossentropy,
    metrics= ["accuracy"],
    )
    
    
    return model





def get_model_2xconv_vary_drop(dropout):
    logging.debug("Model with dropout {}".format(dropout))
    
    model = ks.models.Sequential()

    model.add(ks.layers.Conv2D(32, (3,3), activation = "relu", input_shape=(150,150,3)))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Conv2D(64, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))  
    
    model.add(ks.layers.Flatten()) # This is just a reshape!
    
    model.add(ks.layers.Dropout(dropout))
    
    model.add(ks.layers.Dense(512,activation="relu"))
    model.add(ks.layers.Dense(1,activation="sigmoid"))

    #model.summary()
    model.compile(   
    optimizer = ks.optimizers.RMSprop(lr=0.0001),
    loss= ks.losses.binary_crossentropy,
    metrics= ["accuracy"],
    )
    
    
    return model



def get_model_simple():
    
    model = ks.models.Sequential()

    model.add(ks.layers.Conv2D(32, (3,3), activation = "relu", input_shape=(150,150,3)))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Conv2D(64, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Conv2D(128, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))
    
    model.add(ks.layers.Conv2D(128, (3,3), activation = "relu"))
    model.add(ks.layers.MaxPooling2D(2,2))
    
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



if __name__ == '__main__':
    pass