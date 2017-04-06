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
        print("Input:")
        imutil.show(X)
        print("Ground Truth:")
        imutil.show(map_to_img(Y, **params))
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

    # Easy Target: A single layer CGRU gets this right away
    # Light up the row and column centered on the cat
    #target = crosshair(cx/SCALE, cy/SCALE, color=0)

    # Easy Target:
    # Light up a cone to the right of the cat
    #target = right_cone(cx, cy)

    # Medium Target: Light up for all pixels that are ABOVE the cat AND RIGHT OF the dog
    # Takes a little more training but one layer figures this out
    #target = up_cone(cx, cy) + right_cone(dx, dy)
    #target = (target > 1).astype(np.float)

    # Medium Target: Light up a fixed-radius circle around the cat
    # The only hard part here is learning to ignore the dog
    #target = circle(cx/SCALE, cy/SCALE, 4)

    # Hard Target: Line from cat to dog
    # This can't be done at distance without two layers
    target = line(dx, dy, cx, cy, **params)

    # Hard Target: Light up the midway point between the cat and the dog
    #target = circle((dx+cx)/2/SCALE, (dy+cy)/2/SCALE, 1, color=1)

    # Hard Target: Light up a circle around the cat BUT
    # with radius equal to the distance to the dog
    #rad = math.sqrt((dx-cx)**2 + (dy-cy)**2)
    #target += circle(cx/SCALE, cy/SCALE, rad/SCALE, color=0)
    #target += circle(dx/SCALE, dy/SCALE, rad/SCALE, color=2)

    # For fun, ALSO draw a blue circle around the cat
    #target += circle(cx/SCALE, cy/SCALE, 4, color=2)
    #target = np.clip(target, 0, 1)

    # Add a little epsilon to stave off dead gradient
    target += .05

    # Gaussian blur to smooth the gradient
    from scipy.ndimage.filters import gaussian_filter
    target = gaussian_filter(target, sigma=1.0)

    target *= 10
    target = np.clip(target, 0, 1)

    return pixels, target


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


def crosshair(x, y, color=0):
    height = width
    Y = np.zeros((height, width, IMG_CHANNELS))
    Y[y,:, color] = 1.0
    Y[:,x, color] = 1.0
    return Y


def circle(x, y, r, color=0):
    width = IMG_WIDTH / SCALE
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
    for t in range(100):
        yi = y0 + (t / 100.) * (y1 - y0)
        xi = x0 + (t / 100.) * (x1 - x0)
        Y[int(yi), int(xi), color] = 1.0
    return Y


def map_to_img(Y, width, **kwargs):
    output = np.zeros((width, width, 3))
    from scipy.misc import imresize
    output[:,:] = imresize(Y[:,:], (width, width))
    output *= Y.max()
    return output
