# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:07:57 2019

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
import pickle
import numpy as np
from imagehash import phash
from math import sqrt
from keras.preprocessing.image import img_to_array, array_to_img
from scipy import ndarray
from skimage import transform
from skimage import util
import skimage as sk 
import random
from keras.layers import Dense, Dropout,SeparableConv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras import layers, Model, Input
from keras.optimizers import Adam
import keras 
from sklearn.preprocessing import OneHotEncoder
import re
from scipy.ndimage import affine_transform
import cv2

from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16 

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

if isfile('humpback-whale-identification/p2h.pickle'):
    with open('humpback-whale-identification/p2h.pickle', 'rb') as f:
        p2h = pickle.load(f)
else:
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
            
# split training data into training and validation

train_img, test_img = train_test_split(np.array(breed_imgs), test_size = 0.2)      
#test_img = validation_img

# Bounding  box
def expand_bb(p):
    p = p[:-4]
    for breed in breed_list:
        if isfile(training_bb+'/'+breed+'/'+p): return training_bb+'/'+breed+'/'+p
    return p

def BoundBox(img):
    file = open(expand_bb(img))
    bb = file.read().split('\n')[-7:-3]
    ptobb = [int(s) for i in range(len(bb)) for s in re.findall('\d+',bb[i])]
    file.close()
    return ptobb
#create a dictionary such that every image has its bounding box values (slow process)
train_p2bb = {img: BoundBox(img) for img in train_img}

def draw_dot(draw, x, y):
    draw.ellipse(((x-5,y-5),(x+5,y+5)), fill='red', outline='red')

def draw_dots(draw, coord):
    draw_dot(draw, coord[0], coord[1])
    draw_dot(draw, coord[0], coord[3])
    draw_dot(draw, coord[2], coord[1])
    draw_dot(draw, coord[2], coord[3])    

def bb_orig_img(img):
    img_arr = Image.open(expand_path(img))
    xmin, ymin, xmax, ymax = train_p2bb.get(img)
    crop_img_arr = img_to_array(img_arr)[ymin:ymax, xmin:xmax,:]
    draw = Draw(img_arr)
    draw_dots(draw, train_p2bb.get(img))
    draw.rectangle(train_p2bb.get(img), outline='red')
    return img_arr, crop_img_arr

def bb_cropped_img(img, resize_shape):
    img_shape = Image.open(expand_path(img)).size
    #img_arr = cv2.imread(expand_path(img))
    #img_small = cv2.resize(img_arr, resize_shape, interpolation=cv2.INTER_CUBIC)
    # Get the scaling factor
    # the scaling factor = (y1/y, x1/x)
    # you have to flip because the image.shape is (y,x) but your corner points are (x,y)
    scale = np.flipud(np.divide(resize_shape, (img_shape[1],img_shape[0])))  
    #use this on to get new top left corner and bottom right corner coordinates
    bb = train_p2bb.get(img)
    top_left = (bb[0], bb[3])
    bottom_right = (bb[2], bb[1])
    new_top_left = np.multiply(top_left, scale)
    new_bottom_right = np.multiply(bottom_right, scale)
    return [min(new_top_left), min(new_bottom_right), max(new_bottom_right), max(new_top_left)]

# Resize all image into target shape
def ResizeImage(img, target_shape = (255,255)):
    img_arr = cv2.imread(expand_path(img))
    img_small = cv2.resize(img_arr, target_shape, interpolation=cv2.INTER_CUBIC)
    return img_small

def toBW(img_arr):
    bw_img = array_to_img(img_arr).convert('L')
    return img_to_array(bw_img)
new_shape = (150,150)

# An example of resize image and transform bounding box accordingly 
bb_img, crop_img = bb_orig_img(train_img[1])
bb_small_img = bb_cropped_img(train_img[1], new_shape)

bb_img.show(); array_to_img(crop_img).show();

sml_img = array_to_img(ResizeImage(train_img[1], new_shape))
sml_img.show()
draw = Draw(sml_img)
draw.rectangle(bb_small_img, outline='red')
sml_img.show()

# Select 4000 bounding boxes from training images, then use them to predict 
# the bounding box of the remaining images
train_x, validation_x = train_img[:2000], train_img[2000:]

# Convert images into Black and white (Slow Process)
train_X = np.zeros(shape = ((len(train_x), new_shape[0], new_shape[1], 1) ))
validation_X = np.zeros(shape = ((len(validation_x), new_shape[0], new_shape[1], 1)))
test_X = np.zeros(shape = ((len(test_img), new_shape[0], new_shape[1], 1)))

train_bb = np.zeros(shape = (len(train_x),4), dtype = 'float32')
for i, p in enumerate(train_x):
    train_bb[i] = bb_cropped_img(p, new_shape)
    train_X[i] = toBW(ResizeImage(p, new_shape)) / 255
for i, p in enumerate(validation_x):
    validation_X[i] = toBW(ResizeImage(p, new_shape)) / 255
for i, p in enumerate(test_img):
    test_X[i] = toBW(ResizeImage(p, new_shape)) / 255

for i, (p, bb) in enumerate(zip(train_X[90:100], train_bb[90:100])):
    n = 10
    img = array_to_img(p)
    draw = Draw(img)
    draw.rectangle(bb, outline = 'red')
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)    
    
# Fit train data to CNN model to predict bounding boxes for validation and test data
def VGG_model():
    VGG = VGG16(weights = 'imagenet', include_top = False)
    input_ = Input(shape = (new_shape[0], new_shape[1], 3))
    base_conv = VGG16(input_)
    
    x = Dropout(0.5)(base_conv)
    x = BatchNormalization()(x)
    
    x = Dense(32, activation = 'relu')(x)
    x = Dense(16, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    x = Dense(8, activation = 'relu')(x)
    output_ = Dense(4, activation = 'linear')(x)
    
    model = Model(input_, output_)
    return model


def build_model(with_dropout=True):
    kwargs     = {'activation':'relu', 'padding':'same'}
    conv_drop  = 0.2
    dense_drop = 0.5
    inp        = Input(shape= (new_shape[0], new_shape[1], 1))

    x = inp

    x = Conv2D(64, (9, 9), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)
    
    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    h = MaxPooling2D(pool_size=(1, int(x.shape[2])))(x)
    h = Flatten()(h)
    if with_dropout: h = Dropout(dense_drop)(h)
    h = Dense(16, activation='relu')(h)
    
    v = MaxPooling2D(pool_size=(int(x.shape[1]), 1))(x)
    v = Flatten()(v)
    if with_dropout: v = Dropout(dense_drop)(v)
    v = Dense(16, activation='relu')(v)
    
    x = Concatenate()([h,v])
    if with_dropout: x = Dropout(0.5)(x)
    
    x = Dense(16, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    x = Dense(8, activation = 'relu')(x)
    x = Dense(4, activation='linear')(x)
    return Model(inp,x)

model = build_model(with_dropout=True)
model.summary()

model.compile(optimizer = Adam(lr = 0.032), loss = 'mean_squared_error')
callback_list = [keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 5),
                 keras.callbacks.ModelCheckpoint(filepath = 'bb_model.h5',
                                                 monitor = 'val_loss',
                                                 save_best_only = True)]
model.fit(train_X, train_bb,
          epochs = 50,
          batch_size = 32,
          callbacks = callback_list,
          validation_split = 0.2, 
          verbose = 2)
model.load_weights('bb_model.h5')
# Show the accuaracy of our prediction
#val_bb = [bb_cropped_img(p, new_shape) for p in validation_x]
#val_loss = model.evaluate(validation_X, val_bb, batch_size = 128, verbose = 1)

for i, (p, img_arr) in enumerate(zip(validation_x[10:16], validation_X[10:16])):
    n = 10
    #rows = (i + 5) // 5 
    #cols = (i + 5) % 5
    img = array_to_img(img_arr).convert('RGB')
    draw = Draw(img)
    true_bb = bb_cropped_img(p, new_shape)  
    pred_bb = model.predict(img_arr.reshape(1, 150,150,1)) 
    draw.rectangle(true_bb, outline = 'green')
    draw.rectangle(pred_bb, outline = 'yellow')
    plt.subplot(2, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)


# Predict bounding box for validation and test images
val_pred_bb = model.predict(validation_X, batch_size = 128)
test_pred_bb = model.predict(test_X, batch_size = 128)

# Extract Bounding box from validation and test images 
# and convert bounding box into same shape
def extract_bb(img_arr, bb):
    crop_img_arr = []
    for i, p in enumerate(img_arr):
        xmin, ymin, xmax, ymax = bb[i]
        crop_img_arr.append(p[ymin:ymax, xmin:xmax,:])
    return crop_img_arr

# extract bounding box of resized images
crop_train_img = extract_bb(train_X, train_bb)
crop_val_img = extract_bb(validation_X, val_pred_bb)
crop_test_img = extract_bb(test_X, test_pred_bb)
# transform extracted bounding box into same shape
crop_train_img = [ResizeImage(img_arr, new_shape) for img_arr in crop_train_img]
crop_val_img   = [ResizeImage(img_arr, new_shape) for img_arr in crop_val_img]
crop_test_img  = [ResizeImage(img_arr, new_shape) for img_arr in crop_test_img]

#Image.open(expand_path('n02090379_4667.jpg'))
#Image.open(expand_path('n02090379_855.jpg'))

# convert images into black & white form


train_labels = [p2b.get(p) for p in train_x]
val_labels = [p2b.get(p) for p in validation_x]
test_labels = [p2b.get(p) for p in test_img]

# Merge train and validation data for X and y
train_data = np.concatenate((train_array, val_array), axis = 0)
train_Y = np.concatenate((train_labels, val_labels), axis = 0)
# Image data augment

def random_rotate(image_array: ndarray):
    random_degree = random.uniform(-30,30)
    return transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)

def random_flip(image_array: ndarray):
    return image_array[:,::-1]

def img_augment(img_array):
    num_imgs = len(img_array)
    num_img2tran = np.random.randint(0, num_imgs, size = num_imgs)
    aug_array = np.zeros(shape = img_array.shape, dtype = 'float32')
    aug_label = []
    available_trans = {'rotate': random_rotate,
                       'noise': random_noise,
                       'flip': random_flip}    
    for i in num_img2tran:
        num_tran = np.random.randint(1,4)
        trans_num = 0
        array = img_array[i]
        while trans_num <= num_tran:
            key = random.choice(list(available_trans))
            array = available_trans[key](array)
            trans_num += 1
        aug_array[i, :,:,:] = array
        aug_label.append(p2b.get(train_img[i]))
    
    return aug_array, aug_label 

aug_imgs, aug_Y = img_augment(train_data) 

concat_imgs = np.concatenate((train_data,aug_imgs), axis = 0)
concat_labels = np.concatenate((train_Y, aug_Y), axis = 0)

# Onehot encoding labels
concat_labels = concat_labels.reshape((len(concat_labels), 1))
test_labels = np.array(test_labels).reshape((len(test_labels), 1))

onehot = OneHotEncoder()
onehot.fit(concat_labels)
train_labels = onehot.transform(concat_labels)
test_labels = onehot.transform(test_labels)

# Build a CNN model
img_input = Input(shape = (150, 150, 1))
x = SeparableConv2D(16, (3,3), padding = 'same')(img_input)
x = SeparableConv2D(16, (3,3), padding = 'same')(x)
x = MaxPooling2D((2,2), padding = 'same')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = SeparableConv2D(32, (3,3), padding = 'same')(x)
x = SeparableConv2D(32, (3,3), padding = 'same')(x)
x = MaxPooling2D((2,2), padding = 'same')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = SeparableConv2D(64, (3,3), padding = 'same')(x)
x = SeparableConv2D(64, (3,3), padding = 'same')(x)
x = MaxPooling2D((2,2), padding = 'same')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = SeparableConv2D(16, (3,3), padding = 'same')(x)
x = SeparableConv2D(16, (3,3), padding = 'same')(x)
x = MaxPooling2D((2,2), padding = 'same')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

Flattenpool = layers.Flatten()(x)
Averagepool = layers.GlobalAveragePooling2D()(x)
Maxpool = layers.GlobalMaxPooling2D()(x)

output = layers.Concatenate(axis = -1)([Flattenpool, Averagepool, Maxpool])
output.shape

output = Dense(256, activation = 'relu')(output)
output = Dropout(0.5)(output)
output = BatchNormalization()(output)

output = Dense(128, activation = 'relu')(output)
output = Dropout(0.5)(output)
output = BatchNormalization()(output)

output = Dense(len(breed_list), activation = 'softmax')(output)
model = Model( img_input, output)
model.summary()

learning_rate = 0.0001
optimizer = Adam(learning_rate)
model.compile(optimizer = optimizer, 
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

Earlystop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2)
checkpoint = keras.callbacks.ModelCheckpoint(monitor = 'val_loss',
                                             save_weights_only = True,
                                             filepath = 'stanford_dogs/model1.h5')

model.fit(train_array, train_labels,
          epochs = 100, 
          batch_size = 1280,
          validation_data = (val_array, val_labels),
          callbacks = [Earlystop, checkpoint])
