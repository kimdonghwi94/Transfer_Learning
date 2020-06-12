import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
directory='/home/kim/kfold/1/train'
test_directory='home/kim/kfold/1/test'
count=30592
img_height,img_width=299,299
batch_size=64
epochs=100
import numpy as np
mode=keras.applications.InceptionV3(weights='imagenet',include_top=False,input_shape=(img_height,img_width,3))
modee=mode.output.shape
normalization=ImageDataGenerator(rescale=1/255)
def imageprocessing(dir,sample_count):
    features = np.zeros(shape=(sample_count, modee[1], modee[2], modee[3]))  # 불러온모델의 마지막부분에 매칭되게 넘파이 제로사이즈로 사이즈 만들어 놓기
    labels = np.zeros(shape=(sample_count, 4))
    generator = normalization.flow_from_directory(dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:

        features_batch = mode.predict(inputs_batch)  # generator로 불러온 데이터를 불러온 모델에 집어넣고 특징을 추출
        features[i * batch_size: (i + 1) * batch_size] = features_batch  # 특징을 추출한것을 위에 features 에 갯수 맞춰서 집어넣기
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch  # 같은 방법으로 라벨링
        i += 1
    return features,labels


input_data,label_data=imageprocessing(directory,count)
#Model=keras.models.Sequential()
#Model.add(layers.Flatten(input_shape=date.shape[1:]))
#Model.add(layers.Dense(256,activation='relu',input_dim=(data[1]*data[2]*data[3])))
#Model.add(layers.Dropout(0.25))
#Model.add(layers.Dense(4,activation='sigmoid'))
#Model.compile(optimizer=optimizers.Adam,loss='categorical_crossentropy',metrics=['acc'])
#Model.fit(train,label,batch_size=batch_size,epochs=epochs)
