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
    language_output = layers.Dense(embed_size)(language)

    # Apply the convolutional layers of VGG16
    from keras.applications.vgg16 import VGG16
    vgg = VGG16(include_top=False)
    for layer in vgg.layers:
        layer.trainable = False

    # Run a pretrained network
    x = vgg(input_img)

    # Broadcast language into every convolutional output
    shape = map(int, x.shape)
    language = layers.RepeatVector(shape[1] * shape[2])(language_output)
    language = layers.Reshape((shape[1], shape[2], embed_size))(language)
    x = layers.Concatenate()([x, language])

    # Statefully scan the image in each of four directions
    x = SpatialCGRU(x, cgru_size_1)
    # Stack another one on there
    x = SpatialCGRU(x, cgru_size_2)

    # Add language output again!
    shape = map(int, x.shape)
    language = layers.RepeatVector(shape[1] * shape[2])(language_output)
    language = layers.Reshape((shape[1], shape[2], embed_size))(language)
    x = layers.Concatenate()([x, language])

    # Upsample and convolve
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)

    # Output an RGB image
    x = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs=[input_img, input_words], outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy', lr=.0001)
    model.summary()
    return model


def train(model, model_filename, batches_per_epoch, **params):
    batch_size = params['batch_size']
    validate = params['validate']

    if model_filename and os.path.exists(model_filename):
        model.load_weights(model_filename)

    while True:
        if validate:
            print("Input a command for the network:")
            user_input = raw_input()
            demo(model, user_input, **params)
            continue
        else:
            demo(model, **params)
        for i in range(batches_per_epoch):
            batch_X, batch_Y = get_batch(**params)
            model.train_on_batch(batch_X, batch_Y)
        if model_filename:
            model.save_weights(model_filename)


def demo(model, user_input=None, **params):
    batch_size = params['batch_size']
    # Data for demo prediction
    X, Y = get_batch(**params)
    if user_input:
        X[1][0] = words.indices(user_input)
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


def example(curriculum_level, **params):
    width = params['width']
    validate = params['validate']

    pixels = np.zeros((width, width, 3))
    rand = lambda: np.random.randint(16, width-16-1)

    cx, cy = rand(), rand()
    pixels[cy-16:cy+16, cx-16:cx+16] = cat

    dx, dy = rand(), rand()
    while cx - 8 < dx < cx + 8 and cy - 8 < dy < cy + 8:
        dx, dy = rand(), rand()
    pixels[dy-16:dy+16, dx-16:dx+16] = dog

    target = np.zeros((width, width, IMG_CHANNELS))

    phrase = random.choice([
        'draw a {} cone {} the {}'.format(random.choice(['red', 'green', 'blue']), random.choice(['above', 'below', 'left of', 'right of']), random.choice(['dog', 'cat'])),
        'draw a {} circle around the {}'.format(random.choice(['red', 'green', 'blue']), random.choice(['dog', 'cat'])),
        'draw a {} cross around the {}'.format(random.choice(['red', 'green', 'blue']), random.choice(['dog', 'cat'])),
        'draw a {} line between the cat and dog'.format(random.choice(['red', 'green'])),
        'draw a {} circle between the cat and dog'.format(random.choice(['red', 'green', 'blue'])),
    ][:curriculum_level])
    if validate:
        phrase = random.choice([
            'draw a blue line between the cat and dog',
        ])

    # Easy Target: A single layer CGRU gets this right away
    # Light up the row and column centered on the cat
    if 'cross' in phrase:
        x, y = (cx, cy) if 'cat' in phrase else (dx, dy)
        color = 0 if 'red' in phrase else 1 if 'green' in phrase else 2
        target += crosshair(x, y, color=color, **params)

    # Hard Target: Line from cat to dog
    if 'line between' in phrase:
        color = 0 if 'red' in phrase else 1 if 'green' in phrase else 2
        target += line(dx, dy, cx, cy, color=color, **params)

    # Hard Target: Light up the midway point between the cat and the dog
    if 'circle between' in phrase:
        color = 0 if 'red' in phrase else 1 if 'green' in phrase else 2
        target = circle((dx+cx)/2, (dy+cy)/2, rmin=1, rmax=16, color=color, **params)

    if 'circle around' in phrase:
        x, y = (cx, cy) if 'cat' in phrase else (dx, dy)
        color = 0 if 'red' in phrase else 1 if 'green' in phrase else 2
        target += circle(x, y, rmin=24, rmax=32, color=color, **params)

    if 'cone above' in phrase:
        x, y = (cx, cy) if 'cat' in phrase else (dx, dy)
        color = 0 if 'red' in phrase else 1 if 'green' in phrase else 2
        target = up_cone(x, y, color, **params)

    if 'cone left' in phrase:
        x, y = (cx, cy) if 'cat' in phrase else (dx, dy)
        color = 0 if 'red' in phrase else 1 if 'green' in phrase else 2
        target = left_cone(x, y, color, **params)

    if 'cone right' in phrase:
        x, y = (cx, cy) if 'cat' in phrase else (dx, dy)
        color = 0 if 'red' in phrase else 1 if 'green' in phrase else 2
        target = right_cone(x, y, color, **params)

    if 'cone below' in phrase:
        x, y = (cx, cy) if 'cat' in phrase else (dx, dy)
        color = 0 if 'red' in phrase else 1 if 'green' in phrase else 2
        target = down_cone(x, y, color, **params)

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



def up_cone(x, y, color=0, **params):
    width = params['width']
    Y = np.zeros((width, width, IMG_CHANNELS))
    for i in range(y, -1, -1):
        left = x - (y - i)
        right = x + (y - i) + 1
        left = max(0, left)
        Y[i, left:right, color] = 1.0
    return Y
    

def right_cone(x, y, color=0, **params):
    width = params['width']
    Y = np.zeros((width, width, IMG_CHANNELS))
    for i in range(x, width):
        bot = y - (i - x)
        top = y + (i - x) + 1
        bot = max(0, bot)
        Y[bot:top, i, color] = 1.0
    return Y


def down_cone(x, y, color=0, **params):
    width = params['width']
    Y = np.zeros((width, width, IMG_CHANNELS))
    for i in range(y, width):
        left = max(0, x - (i - y))
        right = min(x + (i - y) + 1, width)
        Y[i, left:right, color] = 1.0
    return Y
    

def left_cone(x, y, color=0, **params):
    width = params['width']
    Y = np.zeros((width, width, IMG_CHANNELS))
    for i in range(0, x):
        bot = y - (x - i)
        top = y + (x - i) + 1
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
        yi, xi = int(yi), int(xi)
        Y[yi-3:yi+3, xi-3:xi+3, color] = 1.0
    return Y


def map_to_img(Y, width, **kwargs):
    output = np.zeros((width, width, 3))
    from scipy.misc import imresize
    output[:,:] = imresize(Y[:,:], (width, width))
    output *= Y.max()
    return output
