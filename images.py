import os
import sys
import tarfile
import zipfile
import numpy as np
from scipy import ndimage
import requests
from urllib.request import urlretrieve



def read_images(archive_link):
    archive_url=requests.get(archive_link,stream=True)
    print(archive_url.headers)
    sys.exit()


def constuct_input_matrix():
    im_path=
    cur_dir=os.listdir();
    if(im_path not in  cur_dir):
        if(im_arc not in  cur_dir):
            tarfile.open(caltech_path+'.tar').extractall()

    for root, subdirs, files in os.walk(caltech_path):
        for f in files:
            print(ndimage.imread(f))
            sys.exit()


if __name__ == '__main__':
    constuct_input_matrix()
    #read_images('http://cs231n.stanford.edu/tiny-imagenet-200.zip')
