import keras
import numpy as np
from keras.preprocessing import image
model = keras.models.load_model('donghwikim/image_model/Resnet_allfine_5')
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
        if predic < 0.5:
            cv2.imwrite('/home/kim/result/False/fifty_under/'+'{},{}.jpg'.format(j[33:],name), save)

        if j[30] == 'N' and prediction == 0:
            cnv.append(1)
            cv2.imwrite('/home/kim/result/Correct/CNV/'+'{},{}.jpg'.format(j[33:],name), save)
        elif j[30] == 'M' and prediction == 1:
            dme.append(1)
            cv2.imwrite('/home/kim/result/Correct/DME/'+'{},{}.jpg'.format(j[33:],name),save)
        elif j[30] == 'R' and prediction == 2:
            drusen.append(1)
            cv2.imwrite('/home/kim/result/Correct/DRUSEN/'+'{},{}.jpg'.format(j[36:],name), save)
        elif j[30] == 'O' and prediction == 3:
            normal.append(1)
            cv2.imwrite('/home/kim/result/Correct/NORMAL/'+'{},{}.jpg'.format(j[36:],name), save)
        elif j[30] == 'N' and prediction != 0:
            cnv.append(0)
            cv2.imwrite('/home/kim/result/False/CNV/'+'{},{}.jpg'.format(j[33:],name), save)
        elif j[30] == 'M' and prediction != 1:
            dme.append(0)
            if prediction==2:
                cv2.imwrite('/home/kim/result/False/DME_DRUSEN/' + '{},{}.jpg'.format(j[33:],name), save)
            elif prediction==3:
                cv2.imwrite('/home/kim/result/False/DME_NORMAL/' + '{},{}.jpg'.format(j[33:],name), save)
        elif j[30] == 'R' and prediction != 2:
            drusen.append(0)
            if prediction==1:
                cv2.imwrite('/home/kim/result/False/DRUSEN_DME/' + '{},{}.jpg'.format(j[36:],name), save)
            elif prediction==3:
                cv2.imwrite('/home/kim/result/False/DRUSEN_NORMAL/' + '{},{}.jpg'.format(j[36:],name), save)
        elif j[30] == 'O' and prediction != 3:
            normal.append(0)
            if prediction==1:
                cv2.imwrite('/home/kim/result/False/NORMAL_DME/' + '{},{}.jpg'.format(j[36:],name), save)
            elif prediction==2:
                cv2.imwrite('/home/kim/result/False/NORMAL_DRUSEN/' + '{},{}.jpg'.format(j[36:],name), save)
        if predic < 0.5:
            if j[30]=='N' or j[30]=='M':
                cv2.imwrite('/home/kim/result/False/fifty_under/'+'{},{}.jpg'.format(j[33:],name), save)
            elif j[30]=='O' or j[30]=='R':
                cv2.imwrite('/home/kim/result/False/fifty_under/' + '{},{}.jpg'.format(j[36:], name), save)

print(cnv)
print(dme)
print(drusen)
print(normal)

print('정확도cnv', np.mean(cnv))
print('정확도dme', np.mean(dme))
print('정확도drusen', np.mean(drusen))
print('정확도normal', np.mean(normal))
l = (np.mean(dme) + np.mean(drusen) + np.mean(normal)) / 3
print('전체', l)

cnv=np.array(cnv)
dme=np.array(dme)
drusen=np.array(drusen)
normal=np.array(normal)
save1=np.save('./CNV',cnv)
save2=np.save('./dme',dme)
save3=np.save('./drusen',drusen)
save4=np.save('./normal',normal)
