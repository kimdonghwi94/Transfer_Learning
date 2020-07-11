import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input,decode_predictions
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import cv2

img=image.load_img("/home/kim/kfold/5/train/CNV/CNV-13823-2.jpeg",target_size=(299,299))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

model = keras.models.load_model('./donghwikim/image_model/InceptionV3_allfine')
pred=model.predict(x)
l=np.argmax(pred)

model.summary()

last_layer=model.get_layer('conv2d_94')

grads= K.gradients(model.output[:,l],last_layer.output)[0]

Flatten=K.mean(grads,axis=(0,1,2))
print(Flatten,Flatten.shape)
iterate = K.function([model.input],[Flatten,last_layer.output[0]])

pool_grad,lask_layer_value=iterate([x])

for i in range(Flatten.shape[0]):
    lask_layer_value[:,:,i]*=pool_grad[i]
heatmap=np.mean(lask_layer_value,axis=-1)


heatmap=np.maximum(heatmap,0)
heatmap/=np.max(heatmap)
plt.imshow(heatmap)
plt.show()

iml=cv2.imread("/home/kim/kfold/5/train/CNV/CNV-13823-2.jpeg")
heatmap=cv2.resize(heatmap,(iml.shape[1],iml.shape[0]))

heatmap=np.uint8(255*heatmap)

heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

super=heatmap*0.2+iml
plt.imshow(super)
plt.show()
cv2.imwrite('./heat.jpg',super)
