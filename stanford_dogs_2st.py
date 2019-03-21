# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:28:51 2019

@author: Ethan
"""


from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras import models
from keras import layers
import os
import numpy as np

training_dir = 'stanford-dogs/images'
breed_list = os.listdir(training_dir)

generator = ImageDataGenerator(rescale = 1/255, 
                               rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=False,
                               fill_mode='reflect',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5],
                               validation_split = 0.2)

from PIL import Image
img = Image.open("stanford-dogs/Images/n02087046-toy_terrier/n02087046_420.jpg")
img.show()
img = img_to_array(img)
img = img.reshape((1,)+ img.shape)
x = generator.flow(img)
array_to_img(x[0].reshape((333,500,3))).show()

train_gen = generator.flow_from_directory(training_dir,
                                          target_size = (150,150),
                                          batch_size = 128,
                                          class_mode = 'categorical',
                                          shuffle = True,
                                          seed = 1,
                                          subset = 'training')
train_gen1 = generator.flow_from_directory(training_dir,
                                          target_size = (150,150),
                                          batch_size = 128,
                                          class_mode = 'categorical',
                                          shuffle = True,
                                          seed = 1,
                                          subset = 'training')
train_gen_amend = train_gen + train_gen1

validation_gen = generator.flow_from_directory(training_dir,
                                               target_size = (150,150),
                                               batch_size = 128,
                                               class_mode = 'categorical',
                                               subset = 'validation',
                                               shuffle = True,
                                               seed = 1)

array_to_img(imgs[8].reshape((150,150,3))).show()
array_to_img(imgs1[8].reshape((150,150,3))).show()

concat_img = np.concatenate((imgs, imgs1))

train_imgs = []
for i in range(len(train_gen)):
    imgs = np.concatenate((list(train_gen[i])[0], list(train_gen1[i])[0]))
    train_imgs.append(imgs)


img_input = Input(shape = (150, 150, 3))
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

model.fit_generator(train_array, train_labels,
          epochs = 100, 
          batch_size = 1280,
          validation_data = (val_array, val_labels),
          callbacks = [Earlystop, checkpoint])
                                        