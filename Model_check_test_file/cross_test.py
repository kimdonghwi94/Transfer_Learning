import numpy as np

import random
from keras.preprocessing import image
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import models
from keras import layers
from keras import optimizers
import keras
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
base_dir = '/home/kim/kfold/5'
test='/home/kim/kfold/5/test/'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

img_width, img_height = 299, 299
train_size=64
conv_base = keras.applications.InceptionResNetV2(weights='imagenet',include_top = False,input_shape=(img_width, img_height, 3))
conv=conv_base.output.shape

#InceptionResNetV2 -95
#Inception-93
#ResNet50V2 -93
#VGG19 - 90

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 64
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, conv[1], conv[2], conv[3]))
    labels = np.zeros(shape=(sample_count,4))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels
train_features, train_labels = extract_features(train_dir, train_size)

epochs = 100
model = models.Sequential()
model.add(layers.Flatten(input_shape=train_features.shape[1:]))
model.add(layers.Dense(256, activation='relu', input_dim=(train_features.shape[1]*train_features.shape[2]*train_features.shape[3])))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(4, activation='softmax'))
model.compile(optimizer=optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['acc'])
history = model.fit(train_features, train_labels,batch_size=batch_size,epochs=1)

categories=['CNV','DME','DRUSEN','NORMAL']
import glob
from keras.preprocessing import image
from PIL import Image
def visualize_predictions(model,real):
    for idx, categorie in enumerate(categories):
        files = glob.glob(test + categorie + "/*")
        for idx2, file in enumerate(files):
            img = Image.open(file)
            img = img.convert('RGB')
            img = img.resize((img_width, img_height))
            data = np.asarray(img)
            data = data.reshape((1, img_width, img_height, 3))
            data = data / 255
            result = model.predict(data)
            result1=real.predict_classes(result)
            print(file)
            print('')
            print(result1)

visualize_predictions(conv_base,model)
