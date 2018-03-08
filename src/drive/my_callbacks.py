'''
Created on Mar 7, 2018

@author: batman
'''

import logging
import keras as ks


#checkpoint = ks.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
class MyLoggingCallback(ks.callbacks.Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, filename, print_fcn=print):
        ks.callbacks.Callback.__init__(self)
        self.print_fcn = print_fcn
        self.filename = filename

    def on_epoch_end(self, epoch, logs={}):
        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        msg += "\n"
        #self.print_fcn(msg)
        #with open()
        with open(self.filename, "a") as myfile:
            myfile.write(msg)



class MyCallback(ks.callbacks.Callback):
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
        

class Test_Callback(ks.callbacks.Callback):
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



if __name__ == '__main__':
    pass