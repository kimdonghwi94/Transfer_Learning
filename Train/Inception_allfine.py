import keras
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
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
img_width, img_height = 240, 299   # 이미지 사이즈 조절
batch_size=16
epochs=50
input=Input(shape=(img_width,img_height,3))
model = keras.applications.DenseNet121(input_tensor=input, include_top=False, weights='imagenet', pooling='max')
x = model.output
x = Dense(512, kernel_initializer='uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(4, activation='softmax', name='softmax')(x)
model = Model(model.input, x)
model.summary()



train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '/home/kim/kfold/5/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
model.compile(loss='categorical_crossentropy',optimizer=optimizers.adam())
history = model.fit_generator(train_generator,epochs=epochs)
model.save('DenseNet121_allfine_5')
#cnn+rnn 모델 sgmentation
#Drunet(dilated - residualu-net)
# pre ilm csi
# 샤페띠
#localization
#클래시피케이션 로컬라이제이션 디텍션 세그맨테이션
#sha1:9a35a97f9a94:19ca16ac017c7c78cdb1c0e45344533b9ebf5779
#