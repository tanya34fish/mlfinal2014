import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import subprocess
from PIL import Image
from scipy.misc import imsave,imresize
from skimage import filter,io
from skimage.feature import hog
from skimage.filter import threshold_otsu
def crop(input):
    input = np.array(input).reshape(122,105)
    B = np.argwhere(input)
    try:
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
        new_input = input[ystart:ystop, xstart:xstop]
    except:
        new_input = input
    return new_input

output = []
input = []
k = 0
with open(sys.argv[1],'r') as f:
    for line in f:
        train = line.strip().split()
        classID = int(train[0])
        #if len(train) < 150:
        #    continue
        output.append(classID)
        #105*122 features
        image = []
        for i in range(12810):
            image.append(0.0)
        for i in range(1,len(train)):
            pixel = train[i].split(':')
            image[int(pixel[0])-1] = float(pixel[1])
        
        im = np.array(image).reshape(122,105)
        thresh = threshold_otsu(im)
        binary = []
        for n in image:
            if n > thresh:
                binary.append(1)
            else:
                binary.append(0)
        #binary = im > thresh
        binary = np.array(binary)
        im = binary.reshape(122,105)
        im = crop(im)
        im = imresize(im,(128,128))
        fd, hog_image = hog(im, orientations=16, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), visualise=True)
        #imsave('otsu/otsu_%d.png' %(k+1), im)
        input.append(fd)
        k += 1
        print(k)
with open(sys.argv[2], 'w') as f:
    for i in range(len(input)):
        classid = output[i]
        f.write('%d' %classid)
        hog_image = input[i]
        for h in hog_image:
            f.write(',%.4f' %h)
        f.write('\n')

