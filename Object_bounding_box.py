# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:17:55 2019

@author: Ethan
"""

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
    base_conv = VGG(input_)
    
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