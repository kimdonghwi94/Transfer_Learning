mport os
import numpy as np
######################################################################################
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#####################################################################################아래코드는 이미지 경로 설정
# base_dir1 = 'C:/Users/우쓰/PycharmProjects/untitled/animal'
base_dir = 'D:/OCT Project/kfold/8'
bas='D:/OCT Project/kfold/8/test'
train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_CNV_dir=os.path.join(train_dir,'CNV')
train_DME_dir=os.path.join(train_dir,'DME')
train_DRUSEN_dir=os.path.join(train_dir,'DRUSEN')
train_NORMAL_dir=os.path.join(train_dir,'NORMAL')

test_CNV_dir=os.path.join(bas,'CNV')
test_DME_dir=os.path.join(bas,'DME')
test_DRUSEN_dir=os.path.join(bas,'DRUSEN')
test_NORMAL_dir=os.path.join(bas,'NORMAL')
    #####################################################################################

    #############################################################################################################################
img_width, img_height = 299, 299   # 이미지 사이즈 조절
train_size, validation_size, test_size=30592,256 , 32 ## 트레인, 밸리데이션, 테스트 이미지 갯수 설정 batch 사이즈와 매칭이되게 ex) batch_size=12 이면 12의배수.
    #############################################################################################################################
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
# from keras.applications import inception_v3  #모델 불러오기
import keras

# conv_base = VGG19(weights='imagenet',
#                     include_top=False,
#                     input_shape=(img_width, img_height, 3))  # 모델을 불러와서 conv_base 에 저장

conv_base = keras.applications.InceptionV3(weights='imagenet',include_top = False,
                        input_shape=(img_width, img_height, 3))

conv=conv_base.output.shape
print(conv_base.output.shape)

# input = Input(shape=(224, 224, 3))
# model = ResNet50(input_tensor=input, include_top=False, weights=None, pooling='max')
# model.summary()
# input = Input(shape=(224, 224, 3))
# model = ResNet50(input_tensor=input, include_top=True, weights=None, pooling='max')
# model.summary()
#input_tensor : 입렵으로 받는 형태.
# include_top : 가장 상단의 fully connected계층들을 포함 시킬지의 여부입니다.
# weight : 케라스에서 미리 pretraining 시켜놓은 weight을 사용 할 것인지 여부입니다.
# pooling : top layer를 포함시키지 않았을때 가장 상단부분의(직관적으로 봤을 때는 하단이지만...)max, avr pooling 방법을 셋팅 할 수 있습니다.

conv_base.summary() #불러온 모델의 구조 보기
    #############################################################################################################################
    # 아래코드는 트레인 혹은 테스트 이미지 불러와서 랜덤으로 시각적으로 무슨이미지인지 확인 하는 코드
import os, random
import matplotlib.pyplot as plt

from keras.preprocessing import image

# def show_pictures(path):
#     random_img = random.choice(os.listdir(path))
#     img_path = os.path.join(path, random_img)
#
#     img = image.load_img(img_path, target_size=(img_width, img_height))
#     img_tensor = image.img_to_array(img)
# Image data encoded as integers in the 0–255 range
#     img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application
#     # plt.imshow(img_tensor)
#     # plt.show()
# for i in range(0, 2):
#     show_pictures(train_cats_dir)
#     show_pictures(train_dogs_dir)
################################################  이미지 normalization 0~1 사이값으로 만들어주기
import os, shutil
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1. / 255)  #0~1사이 값으로 만들어주는 datagen 으로 설정후 아래에서 활용
batch_size = 64  #배치 사이즈

    #########################################################################################################################
def extract_features(directory, sample_count): #데이터를 불러온 모델이 넣기
    features = np.zeros(shape=(sample_count, conv[1], conv[2], conv[3]))  #불러온모델의 마지막부분에 매칭되게 넘파이 제로사이즈로 사이즈 만들어 놓기
        #                           sample 갯수, 7*7*512 행열
    labels = np.zeros(shape=(sample_count,4))
        #                      이미지 라벨링 sample갯수, 4개로 분류

    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='categorical')
        #                                   전처리 코드-- directory 데이터 path, (사이즈) , 몇개씩불러올것인지_batch, 카테고리로컬로 여러개의 분류
        #
    i = 0
    for inputs_batch, labels_batch in generator:
        print(inputs_batch.shape,i)
        features_batch = conv_base.predict(inputs_batch)   #generator로 불러온 데이터를 불러온 모델에 집어넣고 특징을 추출
        features[i * batch_size : (i + 1) * batch_size] = features_batch # 특징을 추출한것을 위에 features 에 갯수 맞춰서 집어넣기
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch # 같은 방법으로 라벨링
        i += 1
        if i * batch_size >= sample_count: # 만들어놓은 array에 갯수가 넘치지않게 설정
            break

    return features, labels

train_features, train_labels = extract_features(train_dir, train_size)  # Agree with our small dataset size
# validation_features, validation_labels = extract_features(test_dir, validation_size)
# test_features, test_labels = extract_features(validation_dir, test_size)

# #########################################################################################################################
#위에서 뽑아낸 추출을 이제 classification 하기위해 마지막 fully connected 레이어 만들어주기
from keras import layers
from keras import optimizers
from time import time
from tensorflow.keras.models import Sequential
epochs = 10000

model = keras.models.Sequential()
model.add(layers.Flatten(input_shape=train_features.shape[1:]))
model.add(layers.Dense(256, activation='relu', input_dim=(train_features.shape[1]*train_features.shape[2]*train_features.shape[3])))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(4, activation='softmax'))
model.summary()

# es = EarlyStopping(patience=10, monitor='val_acc')

model.compile(optimizer=optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['acc'])
history = model.fit(train_features, train_labels,batch_size=batch_size,epochs=epochs)
# model.save('dogs_cat_fcl.h5')

# #########################################################################################################################
#학습된 모델에 대하여 test 셋 넣어서 확인
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
        # 학습 테스트 결과 보기
        try:
            prediction = classifier.predict_classes(features)
        except:
            prediction = classifier.predict_classes(features.reshape(1, train_features.shape[1]*train_features.shape[2]*train_features.shape[3]))
        #plt.imshow(img_tensor)  # 학습한 데이터 실제 이미지 확인
        #plt.show()
        #4개의 어레이를 만들어 정확도 확인  a는CNV, b는DME, c는DRUSEN, d는normal
        print(img_path)
        if prediction==0 and img_path[29] == 'N':
            a.append(1)
        elif prediction==1 and img_path[29] == 'M':
            b.append(1)
        elif prediction== 2 and img_path[29] == 'R':
            c.append(1)
        elif prediction == 3 and img_path[29] == 'O':
            d.append(1)
        elif prediction==0 and img_path[29] != 'N':
            a.append(0)
        elif prediction==1 and img_path[29] != 'M':
            b.append(0)
        elif prediction==2 and img_path[29] != 'R':
            c.append(0)
        elif prediction==3 and img_path[29] != 'O':
            d.append(0)
        

    print('CNV {} , DME {} , DRUSEN {} , NORMAL {}'.format(a,b,c,d))
    print('')
    print('CNV 평균 {}, DME 평균 {} , DRUSEN 평균 {} , NORMAL 평균 {}'.format(np.mean(a),np.mean(b),np.mean(c),np.mean(d)))
    l=(np.mean(a)+np.mean(b)+np.mean(c)+np.mean(d)) / 4
    print(l)


visualize_predictions(model, 3400)  # 테스트할 이미지들의 갯수 설정가능
