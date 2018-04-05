'''
Created on Mar 8, 2018

@author: batman
'''
#import keras as ks
import logging
import os
#import keras.preprocessing.image
from . import my_generators
import time
import pandas as pd
import numpy as np

import sklearn as sk
import sklearn.metrics

def get_metrics(test_df):    
    metrics = dict()
    metrics['accuracy_score'] = sklearn.metrics.accuracy_score(test_df['label'], 
                                    test_df['label_pred'], 
                                    normalize=True,
                                    sample_weight=None)
    
    metrics['roc_auc_score'] = sklearn.metrics.roc_auc_score(y_true = test_df['label'], 
                                                  y_score = test_df['prediction_prob'], 
                                                  average='macro', 
                                                  sample_weight=None)
    
    metrics['confusion_matrix']  = sklearn.metrics.confusion_matrix(test_df['label'], 
                                                    test_df['label_pred'])
    
    metrics['f1_score']  = sklearn.metrics.f1_score(y_true = test_df['label'], 
                                    y_pred = test_df['label_pred'], 
                                    labels=None, 
                                    pos_label=1, 
                                    average='binary', 
                                    sample_weight=None)
    
    
    metrics['log_loss']  = sklearn.metrics.log_loss(y_true = test_df['label'], 
                                        y_pred = test_df['label_pred'],  
                                        eps=1e-15, 
                                        normalize=True, 
                                        sample_weight=None, 
                                        labels=None)
    
    
    metrics['precision_score'] = sklearn.metrics.precision_score(y_true = test_df['label'], 
                                        y_pred = test_df['label_pred'], 
                                                 labels=None, 
                                                 pos_label=1, 
                                                 average='binary', 
                                                 sample_weight=None)

    logging.info("Returning metrics on dataframe".format())

    
    return metrics
    #logging.info("accuracy_score {}".format(accuracy_score))
    #logging.info("roc_auc_score {}".format(roc_auc_score))
    #logging.info("confusion_matrix {}".format(confusion_matrix))
    #logging.info("f1_score {}".format(f1_score))
    #logging.info("log_loss {}".format(log_loss))
    #logging.info("precision_score {}".format(precision_score))





def test_model(model,test_dir_root):
    
    test_generator = my_generators.get_test_generator(test_dir_root,50)
    assert not test_generator.shuffle
    
    num_batches = len(test_generator)
    num_files = test_generator.n
    batch_size = test_generator.batch_size

    t0 = time.time()
    
    # Return a 2D array with floating point probability 0-1
    prediction_probabilities = model.predict_generator(test_generator, 
                                                       verbose=1,
                                                       workers=-2,
                                                       use_multiprocessing=True)
    
    # Convert to 1D array
    prediction_probabilities = prediction_probabilities[:,0]
    
    # Initialize a new array for the TF values
    predictionsTF = np.empty([len(prediction_probabilities)])
    
    # Convert from 2D array to a list
    #prediction_prob = [i[0] for i in prediction_probabilities]
    #predictionsTF = np.array(prediction_prob)
    predictionsTF = prediction_probabilities>=0.5
    predictionsTF = predictionsTF.astype(int)
    
    #predictionsTF[prediction_probabilities>=0.5] = 1
    #predictionsTF[predictionsTF<0.5] = 0
    #predictionsTF = predictionsTF.astype(int)

    #predictionsTF = predictionsTF[predictionsTF == False] = 0
    df = pd.DataFrame({'label':test_generator.classes,
                       'label_pred':predictionsTF,
                       'prediction_prob':prediction_probabilities})

    #print(predictions)
    #print(test_generator.class_indices)
            
    t1 = time.time()
    total = t1-t0
    logging.info("Processed {} images in {} batches. Elapsed time: {}".format(num_files, 
                                                                                num_batches, 
                                                                                total))
    
    return df




if __name__ == '__main__':
    pass
