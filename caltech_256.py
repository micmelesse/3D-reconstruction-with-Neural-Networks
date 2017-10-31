import os
import urllib
import tarfile
import numpy as np

def read_256():
    caltech_path='256_ObjectCategories';
    cur_dir=os.listdir();
    if(caltech_path not in  cur_dir):
        if(caltech_path+'.tar' not in  cur_dir):
            urllib.urlopen('http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar')
        tarfile.open(caltech_path+'.tar').extractall()

    for root, subdirs, files in os.walk(caltech_path):
        for f in files:
            print (f)
