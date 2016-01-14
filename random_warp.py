from skimage import data
from skimage import transform as tf
import numpy as np
import math
from scipy.misc import imsave,imresize
import pickle
import random

def crop(input):
    input = np.array(input).reshape(122,105)
    B = np.argwhere(input)
    try:
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
        new_input = input[ystart:ystop, xstart:xstop]
    except:
        new_input = input
    return new_input

k = 0
output = []
input = []
with open('ml14fall_train.dat' ,'r') as f:
    for line in f:
        train = line.strip().split()
        classID = int(train[0])
        if len(train) < 200:
            k += 1
            continue
        output.append(classID)
        #105*122 features
        image = []
        for i in range(12810):
            image.append(0.0)
        for i in range(1,len(train)):
            pixel = train[i].split(':')
            image[int(pixel[0])-1] = float(pixel[1])
        im = np.array(image).reshape(122,105)
        #im = crop(im)
        #im = imresize(im,(128,128))
        angle = random.randint(8,20)
        locat_y = random.randint(-20,-10)
        rand_scale=random.uniform(0.9,1.2)
        tform = tf.SimilarityTransform(scale=rand_scale, rotation=math.pi / angle,
                                   translation=(im.shape[0] / 20, locat_y) )
        rotated = tf.warp(im, tform)
        #back_rotated = tf.warp(im, tform.inverse)
        #im = crop(im)
        #im = imresize(im,(128,128))
        #im = Image.fromarray(im)
        imsave('random_warp/warp_%d_%d.png' %((k+1),classID), rotated)
        #new = plt.imshow(back_rotated)
        #plt.show()
        k += 1
        print(k)
