import keras
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

img_width, img_height = 299, 299   # 이미지 사이즈 조절
batch_size=32
epochs=50
input=Input(shape=(img_width,img_height,3))
model = keras.applications.InceptionV3(input_tensor=input, include_top=False, weights='imagenet', pooling='max')
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
history = model.fit_generator(train_generator,epochs=epochs,)
model.save('ResNet101_allfine')
