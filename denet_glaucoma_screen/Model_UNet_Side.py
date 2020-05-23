from __future__ import absolute_import
from __future__ import print_function

from tensorflow.python.keras.layers import (Input, Conv2D, MaxPooling2D, Dense,
                                            AveragePooling2D, Flatten)
from tensorflow.python.keras.models import Model


def DeepModel(size_set=640):
    img_input = Input(shape=(size_set, size_set, 3))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(conv5)

    x = AveragePooling2D((7, 7), name='avg_pool')(conv5)

    x = Flatten(name='out_feat')(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dense(2, activation='softmax', name='fc3')(x)

    return Model(img_input, x)
