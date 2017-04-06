import os
import sys
import math
import numpy as np
from keras import layers, models
from PIL import Image
import tensorflow as tf
from keras import backend as K

import imutil
from cgru import SpatialCGRU, transpose, reverse


# Output is RGB
IMG_CHANNELS = 3


def main(**params):
    model = build_model(**params)
    train(model, **params)


def build_model(width, cgru_size_1, cgru_size_2, **params):
    batch_size = params['batch_size']

    # NOTE: All inputs are square
    img = layers.Input(batch_shape=(batch_size, width, width, IMG_CHANNELS))

    # Apply the convolutional layers of VGG16
    from keras.applications.vgg16 import VGG16
    vgg = VGG16(include_top=False)
    for layer in vgg.layers:
        layer.trainable = False

    # Run a pretrained network
    x = vgg(img)

    # Statefully scan the image in each of four directions
    x = SpatialCGRU(x, cgru_size_1)
    # Stack another one on there
    x = SpatialCGRU(x, cgru_size_2)

    # Upsample and convolve
    x = layers.UpSampling2D((2,2))(x)
    # Output an RGB image
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    # Upsample and convolve
    x = layers.UpSampling2D((2,2))(x)
    # Output an RGB image
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    # Upsample and convolve
    x = layers.UpSampling2D((2,2))(x)
    # Output an RGB image
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    # Upsample and convolve
    x = layers.UpSampling2D((2,2))(x)
    # Output an RGB image
    x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
    # Upsample and convolve
    x = layers.UpSampling2D((2,2))(x)
    # Output an RGB image
    x = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

    moo = models.Model(inputs=img, outputs=x)
    moo.compile(optimizer='adam', loss='mse')
    moo.summary()
    return moo


def train(model, model_filename, batches_per_epoch, **params):
    batch_size = params['batch_size']
    X, Y = example(**params)
    print("Input X:")
    imutil.show(X)

    print("Target Y:")
    imutil.show(map_to_img(Y, **params))

    if model_filename and os.path.exists(model_filename):
        model.load_weights(model_filename)

    while True:
        # Data for demo prediction
        examples = [example(**params) for _ in range(batch_size)]
        batch_X, batch_Y = map(np.array, zip(*examples))

        # Predict
        preds = model.predict(batch_X)[-1]
        X = batch_X[-1]
        Y = batch_Y[-1]
        #print("Input:")
        #imutil.show(X)
        #print("Ground Truth:")
        #imutil.show(map_to_img(Y, **params))
        print("Network Output:")
        imutil.show(X + map_to_img(preds, **params))

        print("Training...")
        # Train for a while
        for i in range(batches_per_epoch):
            examples = [example(**params) for _ in range(batch_size)]
            batch_X, batch_Y = map(np.array, zip(*examples))
            h = model.train_on_batch(np.array(batch_X), np.array(batch_Y))

        if model_filename:
            model.save_weights(model_filename)


cat = np.array(Image.open('kitten.jpg').resize((32,32)))
dog = np.array(Image.open('puppy.jpg').resize((32,32)))

def example(**params):
    width = params['width']
    pixels = np.zeros((width, width, 3))
    rand = lambda: np.random.randint(16, width-16-1)
    cx, cy = rand(), rand()
    pixels[cy-16:cy+16, cx-16:cx+16] = cat
    dx, dy = rand(), rand()
    pixels[dy-16:dy+16, dx-16:dx+16] = dog

    target = np.zeros((width, width, IMG_CHANNELS))

    # Easy Target: A single layer CGRU gets this right away
    # Light up the row and column centered on the cat
    #target += crosshair(cx, cy, color=0, **params)

    # Easy Target:
    # Light up a cone to the right of the cat
    #target += right_cone(cx, cy)

    # Medium Target: Light up for all pixels that are ABOVE the cat AND RIGHT OF the dog
    # Takes a little more training but one layer figures this out
    #target += up_cone(cx, cy) + right_cone(dx, dy)
    #target += (target > 1).astype(np.float)

    # Medium Target: Light up a fixed-radius circle around the cat
    # The only hard part here is learning to ignore the dog
    #target += circle(cx, cy, 4)

    # Hard Target: Line from cat to dog
    # This can't be done at distance without two layers
    #target += line(dx, dy, cx, cy, color=1, **params)

    # Hard Target: Light up the midway point between the cat and the dog
    #target = circle((dx+cx)/2, (dy+cy)/2, 1, color=1)

    # Hard Target: Light up a circle around the cat BUT
    # with radius equal to the distance to the dog
    #rad = math.sqrt((dx-cx)**2 + (dy-cy)**2)
    #target += circle(cx, cy, rad, color=0)
    #target += circle(dx, dy, rad, color=2)

    # For fun, ALSO draw a blue circle around the cat
    for r in range(30, 34):
        target += circle(cx, cy, r, color=2, **params)

    target = smooth_gradient(target, **params)
    return pixels, target


def smooth_gradient(target, smoothing_epsilon=.05, max_filter_size=4, **params):
    # Blur to reward approximate correct answers
    if max_filter_size > 1:
        from scipy.ndimage.filters import maximum_filter
        for c in range(IMG_CHANNELS):
            target[:,:,c] = maximum_filter(target[:,:,c], size=max_filter_size)

    # Add a little epsilon to stave off dead gradient
    target += smoothing_epsilon
    return np.clip(target, 0, 1)



def up_cone(x, y, width, color=0, **params):
    Y = np.zeros((width, width, IMG_CHANNELS))
    pos_y, pos_x = int(y * scale), int(x * scale)
    for i in range(pos_y, -1, -1):
        left = pos_x - (pos_y - i)
        right = pos_x + (pos_y - i) + 1
        left = max(0, left)
        Y[i, left:right, color] = 1.0
    return Y
    

def right_cone(x, y, width, color=0, **params):
    Y = np.zeros((width, width, IMG_CHANNELS))
    pos_y, pos_x = int(y * scale), int(x * scale)
    for i in range(pos_x, width):
        bot = pos_y - (i - pos_x)
        top = pos_y + (i - pos_x) + 1
        bot = max(0, bot)
        Y[bot:top, i, color] = 1.0
    return Y


def crosshair(x, y, width, color=0, **params):
    height = width
    Y = np.zeros((height, width, IMG_CHANNELS))
    Y[y,:, color] = 1.0
    Y[:,x, color] = 1.0
    return Y


def circle(x, y, r, width, color=0, **params):
    height = width
    Y = np.zeros((height, width, IMG_CHANNELS))
    for t in range(628):
        yi = y + r * math.cos(t / 100.)
        xi = x + r * math.sin(t / 100.)
        if 0 <= yi < height and 0 <= xi < width:
            Y[int(yi), int(xi), color] = 1.0
    return Y


def line(x0, y0, x1, y1, color=0, **params):
    width = params['width']
    Y = np.zeros((width, width, IMG_CHANNELS))
    dist_scale = abs(x1-x0) + abs(y1-y0)
    for t in range(dist_scale):
        yi = y0 + (t / float(dist_scale)) * (y1 - y0)
        xi = x0 + (t / float(dist_scale)) * (x1 - x0)
        Y[int(yi), int(xi), color] = 1.0
    return Y


def map_to_img(Y, width, **kwargs):
    output = np.zeros((width, width, 3))
    from scipy.misc import imresize
    output[:,:] = imresize(Y[:,:], (width, width))
    output *= Y.max()
    return output
