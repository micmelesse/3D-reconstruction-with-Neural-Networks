import os
import sys
import requests
import tarfile
import zipfile
import numpy as np
from PIL import Image

from urllib.request import urlretrieve



def read_images(archive_link):
    archive_url=requests.get(archive_link,stream=True)
    print(archive_url.headers)
    sys.exit()

    cur_dir=os.listdir();
    if(im_path not in  cur_dir):
        if(im_arc not in  cur_dir):
            tarfile.open(im_path+'.tar').extractall()


def constuct_input_matrix(im_dir="ImageNet"):
    ret=[];
    for root, subdirs, files in os.walk(im_dir):
        for f in files:
            try:
                im=np.array(Image.open(root+'/'+f));
                if im.ndim is 3:
                    im.resize((255,255,3))
                    ret.append(im);
            except OSError:
                pass;

    return np.stack(ret)


# if __name__ == '__main__':
    #     constuct_input_matrix()
    #read_images('http://cs231n.stanford.edu/tiny-imagenet-200.zip')
