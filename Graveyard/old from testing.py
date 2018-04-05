 
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
