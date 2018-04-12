import os
import re
import shutil
import logging 

logging.basicConfig(level=logging.DEBUG)



path_root = r"/home/batman/d3/data/tub_67_18-04-11"
tgt_path = r"/home/batman/d2/data/tub_67/imgs"
assert os.path.exists(path_root)
assert os.path.exists(tgt_path)```


files = os.listdir(path_root)

ext_str = ".jpg"
zero_padding = 7
jpg_files = [fname for fname in files if os.path.splitext(fname)[1] == ext_str]


logging.info("\t{} {} ext_str files in {}".format(len(jpg_files),ext_str,path_root))

#these_files = jpg_files[0:10]
these_files = jpg_files

#this_file = these_files[0]


#print(this_file)
#this_num = re.findall("\d+",this_file)[0]
#print(this_num)
#print(this_num.zfill(6))
#print(this_num.zfill(7)+'_cam-image_array_.jpg')

for this_file in these_files:
    path_source = os.path.join(path_root,this_file)
    
    this_num = re.findall("\d+",this_file)[0]
    new_name = this_num.zfill(zero_padding)+'_cam-image_array_.jpg'
    path_tgt= os.path.join(tgt_path,new_name)
    print(path_source,'\t',path_tgt)
    
    #raise
    shutil.copy(path_source, path_tgt)
    
#_cam-image_array_.jpg
#4853



#re.mat

#print(jpg_files[0])
#print(os.path.splitext(files[0]))