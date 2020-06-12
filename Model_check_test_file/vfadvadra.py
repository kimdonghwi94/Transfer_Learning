import keras
import numpy as np
from keras.preprocessing import image
model = keras.models.load_model('donghwikim/image_model/Xception_allfine_5')
import glob


a='/home/kim/OCT_IMAGE/Crop_OCT/'
b=['DME','DRUSEN','NORMAL']
cnv=[]
dme=[]
drusen=[]
normal=[]
import cv2
for i in b:
    file=glob.glob(a+i+'/*')
    for j in file:
        img = image.load_img(j, target_size=(299, 299))
        img_tensor = np.array(img)
        img_tensor = img_tensor.reshape((1, 299, 299, 3))
        img_tensor = img_tensor / 255
        good = model.predict(img_tensor)
        goods=good[0]
        name=np.round(goods,3)
        predic=np.max(good)
        prediction = np.argmax(good)
        # print("결과", prediction)

        # print('실제값', j)
        save=cv2.imread(j)
        if prediction==0:
            if j[30]=='M':
                cv2.imwrite('/home/kim/result_XCEPTION/False/DME_CNV/' + '{},{}.jpg'.format(j[33:], name), save)
            elif j[30]=='R':
                cv2.imwrite('/home/kim/result_XCEPTION/False/DRUSEN_CNV/' + '{},{}.jpg'.format(j[36:], name), save)
            elif j[30]=='O':
                cv2.imwrite('/home/kim/result_XCEPTION/False/NORMAL_CNV/' + '{},{}.jpg'.format(j[36:], name), save)
