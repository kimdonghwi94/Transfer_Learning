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
test='/home/kim/kfold/5/test'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

train_CNV_dir=os.path.join(train_dir,'CNV')
train_DME_dir=os.path.join(train_dir,'DME')
train_DRUSEN_dir=os.path.join(train_dir,'DRUSEN')
train_NORMAL_dir=os.path.join(train_dir,'NORMAL')

test_CNV_dir=os.path.join(test,'CNV')
test_DME_dir=os.path.join(test,'DME')
test_DRUSEN_dir=os.path.join(test,'DRUSEN')
test_NORMAL_dir=os.path.join(test,'NORMAL')

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
train_size = 30592
conv_base = keras.applications.InceptionV3(weights='imagenet',include_top = False,input_shape=(img_width, img_height, 3))
conv=conv_base.output.shape

conv_base.summary()
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
        print(inputs_batch.shape,i)
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
model.summary()
model.compile(optimizer=optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['acc'])
history = model.fit(train_features, train_labels,batch_size=batch_size,epochs=epochs)
model.save('nonfine_InceptionV3_5')

a=[]
b=[]
c=[]
d=[]
def visualize_predictions(classifier, n_cases):
    for i in range(0,n_cases):
        path = random.choice([test_CNV_dir,test_DME_dir,test_DRUSEN_dir,test_NORMAL_dir]) #테스트데이터 경로 랜덤으로 선택
        random_img = random.choice(os.listdir(path))#
        img_path = os.path.join(path, random_img)# 테스트 이미지 랜덤으로 선택
        img = image.load_img(img_path, target_size=(img_width, img_height)) #랜덤으로 가져온 데이터 전처리
        img_tensor = image.img_to_array(img)
        img_tensor /= 255. #데이터 0~1사이값으로 전처리
        features = conv_base.predict(img_tensor.reshape(1,img_width, img_height, 3))
        try:
            prediction = classifier.predict_classes(features)
        except:
            prediction = classifier.predict_classes(features.reshape(1, train_features.shape[1]*train_features.shape[2]*train_features.shape[3]))
        print(img_path)
        if prediction==0 and img_path[24] == 'N':
            a.append(1)
        elif prediction==1 and img_path[24] == 'M':
            b.append(1)
        elif prediction== 2 and img_path[24] == 'R':
            c.append(1)
        elif prediction == 3 and img_path[24] == 'O':
            d.append(1)
        elif prediction==0 and img_path[24] != 'N':
            a.append(0)
        elif prediction==1 and img_path[24] != 'M':
            b.append(0)
        elif prediction==2 and img_path[24] != 'R':
            c.append(0)
        elif prediction==3 and img_path[24] != 'O':
            d.append(0)
    print('CNV {} , DME {} , DRUSEN {} , NORMAL {}'.format(a,b,c,d))
    print('')
    print('CNV 평균 {}, DME 평균 {} , DRUSEN 평균 {} , NORMAL 평균 {}'.format(np.mean(a),np.mean(b),np.mean(c),np.mean(d)))
    l=(np.mean(a)+np.mean(b)+np.mean(c)+np.mean(d)) / 4
    print(l)
# visualize_predictions(model, 3400)  # 테스트할 이미지들의 갯수 설정가능
