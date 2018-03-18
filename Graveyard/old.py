def get_weights(this_run_path):
    # Get saved weights
    weights_files = [(f,os.path.join(this_run_path,f)) for f in os.listdir(this_run_path) 
                     if re.match('weights-',f)]
    length = len([i for i in weights_files])

    
    # Add the sizes of the weights
    sizes = [os.path.getsize((os.path.join(this_run_path,f)))/1024/1024 
            for f in os.listdir(this_run_path) 
            if re.match('weights-',f)]
    
    total_size = sum(sizes)
    if not total_size == 0:
        avg_size = sum(sizes)/length
    else:
        avg_size = 0
    
    # Add the number of the epoch
    #wts = analysis.get_weights(this_run_path)
    #new_wts = 
    for wt in wts:
        wt = list(wt)
        wt_name = wt[0][0]
        #print(wt_name)
        epoch_num = re.findall(r'epoch\d+', wt_name)
        epoch_num = re.findall(r'\d+', epoch_num[0])
        wt.append(epoch_num)
        
    
    logging.debug("Found {} weights files, total {:.0f} MB = {:.1f} MB per file".format(
        length,total_size,avg_size
        )
    )
    
    weights_files = zip(weights_files,sizes)
    return weights_files

root_path = os.path.join(this_project_path,project_name)
run_folders = [dir for dir in os.listdir(root_path) if re.match('run',dir)]
run_folders.sort()
for rfolder in run_folders:
    this_run_path = os.path.join(root_path,rfolder)
    print('RUN ****', rfolder)

    ###### Weights ######
    analysis.get_weights(this_run_path)    
    
    ###### Architecture ######
    path_arch = analysis.get_architecture(this_run_path)
    print('arch',arch)
    from keras.models import model_from_json    
    with open(path_arch,'r') as arch_file:
        arch_dict = json.load(arch_file)
    with open(path_arch,'r') as arch_file:
        loaded_model_json = arch_file.read()
        model = model_from_json(loaded_model_json)
    model.summary
    #print(arch_dict)
    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    
    raise
    for layer in arch_dict['config']:
        #this_layer = arch_dict['config'][k]
        pprint(layer)
        #print(layer['class_name'])
        layer_str = layer_funcs[layer['class_name']](layer)
        print(layer_str)

    raise
    ###### History ######
    path_hist = analysis.get_history(this_run_path)    
    #print('hist',path_hist)
    with open(path_hist) as hist_file:
        hist_dict = json.load(hist_file)
    #print(hist_dict)
    print("Epochs",hist_dict['params']['epochs'])
    print("Steps",hist_dict['params']['steps'])
    
    plot_hist(hist_dict)
    #raise


def get_weights(this_run_path):
    """Return the weights hdf5 files from a directory
    Creates a sorted dictionary (first to last)
    epoch
    fname
    path
    size
    """
    # File name
    wts_file_name = [f for f in os.listdir(this_run_path)
                     if re.match('weights-',f)]
    
    # Total number of files
    length = len([i for i in wts_file_name])
    
    # File path
    wts_file_path = [os.path.join(this_run_path,f) for f in os.listdir(this_run_path) 
                     if re.match('weights-',f)]
    
    # File sizes
    sizes = [os.path.getsize((os.path.join(this_run_path,f)))/1024/1024 
            for f in os.listdir(this_run_path) 
            if re.match('weights-',f)]

    total_size = sum(sizes)
    if not total_size == 0:
        avg_size = sum(sizes)/length
    else:
        avg_size = 0
    
    # Epoch numbers
    epochs = [re.findall(r'epoch\d+', f)[0] for f in wts_file_name]
    epoch_num = [int(re.findall(r'\d+', f)[0]) for f in epochs]
    #print(epoch_num)



    weights_files = list(zip(wts_file_name,wts_file_path,sizes,epoch_num))
    
    # Sort
    weights_files.sort(key=lambda tup: tup[3])
    
    # Convert to dict
    wt_dicts = list()
    wt_dicts = [{'epoch':i[3],'fname':i[0], 'path':i[1],'size':i[2]} for i in weights_files] 

    logging.debug("Found {} weights files, total {:.0f} MB = {:.1f} MB per file".format(
        length,total_size,avg_size
        )
    )    
    
    return wt_dicts



def old_test_gen():
    
    %%script false
    test_generator = test_datagen.flow_from_directory(
            r'/home/batman/Dropbox/DATA/cats_dogs_all/SMALL_test',
            target_size = (150,150),
            batch_size = 5,
            shuffle=False,
            #class_mode = "binary",
        );
    #print(len(test_generator))
    
    #print(test_generator.n)
    
    #for i in dir (test_generator):
    #    print(i)
    batches = list()
    num_batches = len(test_generator)
    seen_files = 0
    num_files = test_generator.n
    
    for i, batch in enumerate(test_generator):
        seen_files += batch[0].shape[0]
        print("seen {} / {}".format(seen_files,num_files))
        
        #print()
        if i+1 == num_batches:
            break
        #raise
        #print(batch[0].shape)
        #batches.append(batch[0].shape)
        #print(batch[0][0].size)
    #print(len(batches))
    print(seen_files)



def asdf():
    import re
    date_re = re.compile('(?P<a_year>\d{2,4})-(?P<a_month>\d{2})-(?P<a_day>\d{2}) (?P<an_hour>\d{2}):(?P<a_minute>\d{2}):(?P<a_second>\d{2}[.\d]*)')
    found = date_re.match('2016-02-29 12:34:56.789')
    if found is not None:
        print found.groupdict()
    #{'a_year': '2016', 'a_second': '56.789', 'a_day': '29', 'a_minute': '34', 'an_hour': '12', 'a_month': '02'}
    found.groupdict()['a_month']
    #'02'
