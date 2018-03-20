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
def test_model(model,data_dict):
    
    #print(model)
    #test_generator = my_generators.get_train_generator_simple(data_dict['train'],batch_size)

    pths = list(os.walk(data_dict['my_test']))
    
    test_generator = my_generators.get_test_generator(data_dict['my_test'],200)

    num_batches = len(test_generator)
    num_files = test_generator.n
    batch_size = test_generator.batch_size

    t0 = time.time()
    
    
    prediction_probabilities = model.predict_generator(test_generator, 
                                                       verbose=1,
                                                       workers=-2,
                                                       use_multiprocessing=True)

    #print(test_generator.classes)
    
    prediction_prob = [i[0] for i in prediction_probabilities]
    predictionsTF = np.array(prediction_prob)
    predictionsTF[predictionsTF>=0.5] = 1
    predictionsTF[predictionsTF<0.5] = 0
    predictionsTF = predictionsTF.astype(int)
    #pd.Series()
    # >= 0.5 
    #predictionsTF = predictionsTF[predictionsTF == True] = 1
    #predictionsTF = predictionsTF[predictionsTF == False] = 0
    df = pd.DataFrame({'label':test_generator.classes,
                       'label_pred':predictionsTF,
                       'prediction_prob':prediction_prob})

    #print(predictions)
    #print(test_generator.class_indices)
            
    t1 = time.time()
    total = t1-t0
    logging.info("Processed {} images in {} batches. Elapsed time: {}".format(num_files, 
                                                                                num_batches, 
                                                                                total))
    
    return df


 
    #print(prediction_probabilities)
    #print(prediction_probabilities[0])
    #raise
    if 0:    
        num_batches = len(test_generator)
        num_files = test_generator.n
        batch_size = test_generator.batch_size
    
        seen_files = 0
        predictions_list = list()
        
        t0 = time.time()
        
        for i,batch in enumerate(test_generator):
            # Tally the actual seen images (tensor layers)
            seen_files += batch[0].shape[0]
            
            # Current index
            idx = test_generator.batch_index
            
            # Report
            logging.info("{} seen {} / {} = {:.1f}%".format(idx,seen_files,num_files,seen_files/num_files*100))
        
            # Make predictions and append
            predictions = model.predict(batch[0])
            predictions = [i[0] for i in predictions]
            predictions_list += predictions
            
            # Seen all batches, break the loop 
            if i+1 == num_batches:
                break
        
        t1 = time.time()
        total = t1-t0
        logging.info("Processed {} images in {} batches. Elapsed time: {}}".format(seen_files, num_batches, total))
        
        #print(len(pths))
        #data_dict['folders'][]
        #data_dict['folders']
        #model
        logging.info("".format())


if __name__ == '__main__':
    pass
