# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:50:56 2019

@author: Ethan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL.ImageDraw import Draw
from os.path import isfile
import numpy as np
from imagehash import phash
from math import sqrt
from keras.preprocessing.image import img_to_array, array_to_img



training_dir = 'stanford-dogs/images'
training_bb = 'stanford-dogs/Annotation'
breed_list = os.listdir(training_dir)

p2b = {}
breed_imgs = []
for breed in breed_list: 
    imgs = os.listdir(training_dir + '/'+breed)
    breed_imgs += imgs
    for img in imgs:
        p2b[img] = breed
    

# Identify same images from same dog breed
def expand_path(p):
    for breed in breed_list:
        if isfile(training_dir+'/'+breed+'/'+p): return training_dir+'/'+breed+'/'+p
    return p

def match(h1,h2):
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
            i1 =  Image.open(expand_path(p1))
            i2 =  Image.open(expand_path(p2))
            if i1.mode != i2.mode or i1.size != i2.size: return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            a1 = a1/sqrt((a1**2).mean())
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2/sqrt((a2**2).mean())
            a  = ((a1 - a2)**2).mean()
            if a > 0.1: return False
    return True



# Compute phash for each image in the training and test set.
p2h = {}
for p in breed_imgs:
    img    = Image.open(expand_path(p))
    h      = phash(img)
    p2h[p] = h

# Find all images associated with a given phash value.
h2ps = {}
for p,h in p2h.items():
    if h not in h2ps: h2ps[h] = []
    if p not in h2ps[h]: h2ps[h].append(p)

# Find all distinct phash values
hs = list(h2ps.keys())

# If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
h2h = {}
for i,h1 in enumerate(hs):
    for h2 in hs[:i]:
        if h1-h2 <= 6 and match(h1, h2):
            s1 = str(h1)
            s2 = str(h2)
            if s1 < s2: s1,s2 = s2,s1
            h2h[s1] = s2

# Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
for p,h in p2h.items():
    h = str(h)
    if h in h2h: h = h2h[h]
    p2h[p] = h

len(p2h), list(p2h.items())[:5]
h2ps = {}
for p,h in p2h.items():
    if h not in h2ps: h2ps[h] = []
    if p not in h2ps[h]: h2ps[h].append(p)


# Remove same pics with smaller resolution

remove_list = []
for h, ps in h2ps.items():
    if len(ps) > 1 : remove_list.append(ps)
        
for ps in remove_list:        
    size0 = Image.open(expand_path(ps[0])).size
    size1 = Image.open(expand_path(ps[1])).size
    p = ps[0] if size0[0]*size0[1] > size1[0]*size1[1] else ps[1]
    breed_imgs.remove(p)
