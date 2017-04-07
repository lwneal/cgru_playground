import random
import os
import sys
import math
import numpy as np
from keras import layers, models
from PIL import Image
import tensorflow as tf
from keras import backend as K

import words
import imutil
from cgru import SpatialCGRU
from words import MAX_WORDS


# Output is RGB
# NOTE: Input images must be square
IMG_CHANNELS = 3

cat = np.array(Image.open('kitten.jpg').resize((32,32)))
dog = np.array(Image.open('puppy.jpg').resize((32,32)))


def main(**params):
    model = build_model(**params)
    train(model, **params)


def build_model(width, cgru_size_1, cgru_size_2, embed_size=256, **params):
    batch_size = params['batch_size']

    input_img = layers.Input(batch_shape=(batch_size, width, width, IMG_CHANNELS))

    input_words = layers.Input(batch_shape=(batch_size, MAX_WORDS), dtype='int32')
    
    language = layers.Embedding(embed_size, words.VOCABULARY_SIZE)(input_words)
    language = layers.GRU(embed_size)(language)
    language = layers.Dense(embed_size)(language)

    # Apply the convolutional layers of VGG16
    from keras.applications.vgg16 import VGG16
    vgg = VGG16(include_top=False)
    for layer in vgg.layers:
        layer.trainable = False

    # Run a pretrained network
    x = vgg(input_img)

    # Broadcast language into every convolutional output
    shape = map(int, x.shape)
    language = layers.RepeatVector(shape[1] * shape[2])(language)
    language = layers.Reshape((shape[1], shape[2], embed_size))(language)

    x = layers.Concatenate()([x, language])

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

    moo = models.Model(inputs=[input_img, input_words], outputs=x)
    moo.compile(optimizer='adam', loss='mse')
    moo.summary()
    return moo


def train(model, model_filename, batches_per_epoch, **params):
    batch_size = params['batch_size']

    if model_filename and os.path.exists(model_filename):
        model.load_weights(model_filename)

    while True:
        demo(model, **params)
        for i in range(batches_per_epoch):
            batch_X, batch_Y = get_batch(**params)
            model.train_on_batch(batch_X, batch_Y)
        if model_filename:
            model.save_weights(model_filename)


def demo(model, **params):
    batch_size = params['batch_size']
    # Data for demo prediction
    X, Y = get_batch(**params)
    preds = model.predict(X)
    print("Target (left) vs. Network Output (right):")
    input_pixels, input_words = X[0][0], X[1][0]
    print(words.words(input_words))
    left = input_pixels + Y[0] * 255.
    right = input_pixels + map_to_img(preds[0], **params)
    imutil.show(np.concatenate((left, right), axis=1))


def get_batch(**params):
    width = params['width']
    batch_size = params['batch_size']
    batch_X_words = np.zeros((batch_size, MAX_WORDS), dtype=int)
    batch_X_pixels = np.zeros((batch_size, width, width, IMG_CHANNELS))
    batch_Y = np.zeros((batch_size, width, width, IMG_CHANNELS))

    for i in range(batch_size):
        input_pixels, input_words, target_pixels = example(**params)
        batch_X_pixels[i] = input_pixels
        batch_X_words[i] = input_words
        batch_Y[i] = target_pixels

    return [batch_X_pixels, batch_X_words], batch_Y


def example(**params):
    width = params['width']
    pixels = np.zeros((width, width, 3))
    rand = lambda: np.random.randint(16, width-16-1)
    cx, cy = rand(), rand()
    pixels[cy-16:cy+16, cx-16:cx+16] = cat
    dx, dy = rand(), rand()
    pixels[dy-16:dy+16, dx-16:dx+16] = dog

    target = np.zeros((width, width, IMG_CHANNELS))

    phrase = random.choice([
        'draw a red cross through the cat',
        'draw a blue circle around the dog',
        'draw a green circle around the cat',
    ])

    # Easy Target: A single layer CGRU gets this right away
    # Light up the row and column centered on the cat
    if 'cross' in phrase:
        x, y = (cx, cy) if 'cat' in phrase else (dx, dy)
        color = 0 if 'red' in phrase else 1 if 'green' in phrase else 2
        target += crosshair(x, y, color=0, **params)

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

    if 'circle' in phrase:
        x, y = (cx, cy) if 'cat' in phrase else (dx, dy)
        color = 0 if 'red' in phrase else 1 if 'green' in phrase else 2
        target += circle(x, y, rmin=24, rmax=32, color=color, **params)

    target = smooth_gradient(target, **params)
    return pixels, words.indices(phrase), target


def smooth_gradient(target, smoothing_epsilon=.05, max_filter_size=1, **params):
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
    Y[y-4:y+4,:, color] = 1.0
    Y[:,x-4:x+4, color] = 1.0
    return Y


def circle(x, y, rmin, rmax, width, color=0, **params):
    height = width
    Y = np.zeros((height, width, IMG_CHANNELS))
    resolution = 100.0
    for t in range(int(3.14 * 2 * resolution)):
        for r in range(rmin, rmax):
            yi = y + r * math.cos(t / resolution)
            xi = x + r * math.sin(t / resolution)
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
