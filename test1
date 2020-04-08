from keras import models, layers
from keras.applications import VGG16
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

img_width, img_height = 240, 240   # 이미지 사이즈 조절

epochs=100
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = os.path.join('C:/Users/우쓰/PycharmProjects/untitled/data/batch2/train_mini')
# val_dir = os.path.join('./dataset/1/images/val')
test_dir = os.path.join('C:/Users/우쓰/PycharmProjects/untitled/data/batch2/test_mini')

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=8, target_size=(img_width, img_height), color_mode='rgb')
# val_generator = val_datagen.flow_from_directory(val_dir, batch_size=16, target_size=(220, 200), color_mode='rgb')

input_tensor = Input(shape=(img_width, img_height, 3), dtype='float32', name='input')

pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
pre_trained_vgg.trainable = False
pre_trained_vgg.summary()

additional_model = models.Sequential()
additional_model.add(pre_trained_vgg)
additional_model.add(layers.Flatten())
additional_model.add(layers.Dense(4096, activation='relu'))
additional_model.add(layers.Dense(2048, activation='relu'))
additional_model.add(layers.Dense(1024, activation='relu'))
additional_model.add(layers.Dense(4, activation='softmax'))

additional_model.summary()

checkpoint = ModelCheckpoint(filepath='pretrained_VGG_weight.hdf5',
                             monitor='loss',
                             mode='min',
                             save_best_only=True)

additional_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

history = additional_model.fit_generator(train_generator,
                                         steps_per_epoch=math.ceil(train_generator.n / train_generator.batch_size),
                                         epochs=10,
                                         callbacks=[checkpoint])
# history = additional_model.fit_generator(train_generator,
#                                          steps_per_epoch=math.ceil(train_generator.n / train_generator.batch_size),
#                                          epochs=epochs,
#                                          validation_data=val_generator,
#                                          validation_steps=math.ceil(val_generator.n / val_generator.batch_size),
#                                          callbacks=[checkpoint])
