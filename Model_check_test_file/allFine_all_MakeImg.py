import keras
import numpy as np
import random
import os
from keras.preprocessing import image
bas='./create/'
test_CNV_dir=os.path.join(bas,'CNV')
test_DME_dir=os.path.join(bas,'DME')
test_DRUSEN_dir=os.path.join(bas,'DRUSEN')
test_NORMAL_dir=os.path.join(bas,'NORMAL')
a=[]
b=[]
c=[]
d=[]
import cv2
def visualize_predictions(classifier, n_cases):
    for i in range(0,n_cases):
        # patha = 'test_CNV_dir'  # 테스트데이터 경로 랜덤으로 선택
        # pathb = 'test_DME_dir'
        # pathc = 'test_DRUSEN_dir'
        # pathd = 'test_NORMAL_dir'
        # random_img = os.listdir(patha)  # \
        #
        # img_path = os.path.join(patha, random_img)  # 테스트 이미지 랜덤으로 선택

        path = random.choice([test_CNV_dir,test_DME_dir,test_DRUSEN_dir,test_NORMAL_dir]) #테스트데이터 경로 랜덤으로 선택
        random_img = random.choice(os.listdir(path))#\

        img_path = os.path.join(path, random_img)# 테스트 이미지 랜덤으로 선택

        img = image.load_img(img_path, target_size=(299, 299)) #랜덤으로 가져온 데이터 전처리

        img_tensor = np.array(img)

        img_tensor=img_tensor.reshape((1,299,299,3))
        img_tensor =img_tensor/ 255 #데이터 0~1사이값으로 전처리
        im = cv2.imread(img_path)

        prediction=classifier.predict(img_tensor)
        prediction=np.argmax(prediction)
        print(img_path)
        if prediction==0 and img_path[10] == 'N':
            a.append(1)
        elif prediction==1 and img_path[10] == 'M':
            b.append(1)
        elif prediction== 2 and img_path[10] == 'R':
            c.append(1)
        elif prediction == 3 and img_path[10] == 'O':
            d.append(1)
        elif prediction == 0 and img_path[24] != 'N':
            a.append(0)
            if img_path[10] == 'M':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/DME/{}'.format(random_img),im)
            elif img_path[10] == 'R':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/DRUSEN/{}'.format(random_img), im)
            elif img_path[10] == 'O':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/NORMAL/{}'.format(random_img), im)
        elif prediction == 1 and img_path[24] != 'M':
            b.append(0)
            if img_path[10] == 'N':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/CNV/{}'.format(random_img),im)
            elif img_path[10] == 'R':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/DRUSEN/{}'.format(random_img), im)
            elif img_path[10] == 'O':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/NORMAL/{}'.format(random_img), im)
        elif prediction == 2 and img_path[24] != 'R':
            c.append(0)
            if img_path[10] == 'M':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/DME/{}'.format(random_img),im)
            elif img_path[10] == 'N':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/CNV/{}'.format(random_img),im)
            elif img_path[10] == 'O':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/NORMAL/{}'.format(random_img), im)
        elif prediction == 3 and img_path[24] != 'O':
            d.append(0)
            if img_path[10] == 'M':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/DME/{}'.format(random_img),im)
            elif img_path[10] == 'R':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/DRUSEN/{}'.format(random_img), im)
            elif img_path[10] == 'N':
                cv2.imwrite('/home/kim/PycharmProjects/tensorflow2.0-gpu/donghwikim/createea/CNV/{}'.format(random_img),im)

    print('CNV {} , DME {} , DRUSEN {} , NORMAL {}'.format(a,b,c,d))
    print('')
    print('CNV 평균 {}, DME 평균 {} , DRUSEN 평균 {} , NORMAL 평균 {}'.format(np.mean(a),np.mean(b),np.mean(c),np.mean(d)))
    l=(np.mean(a)+np.mean(b)+np.mean(c)+np.mean(d)) / 4
    print(l)
model = keras.models.load_model('image_model/OCTmodel5')

visualize_predictions(model, 303)  # 테스트할 이미지들의 갯수 설정가능
