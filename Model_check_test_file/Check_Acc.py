import keras
import numpy as np
from keras.preprocessing import image
model = keras.models.load_model('donghwikim/image_model/Xception_allfine_5')
import glob

a=glob.glob('/home/kim/OCT/*')
cnv=[]
dme=[]
drusen=[]
normal=[]
for i in a:
    b=glob.glob(i+'/*')
    for j in b:
        c=glob.glob(j+'/*')
        for l in c:
            d=glob.glob(l+'/*')
            for m in d:
                e=glob.glob(m+'/*')
                for z in e:
                    
                    img=image.load_img(z,target_size=(299, 299))
                    img_tensor = np.array(img)
                    img_tensor = img_tensor.reshape((1, 299, 299, 3))
                    img_tensor = img_tensor / 255
                    good=model.predict(img_tensor)
                    prediction = np.argmax(good)
                    print("결과",prediction)

                    print('실제값',z)
                    if z[15] == 'N' and prediction==0:
                        cnv.append(1)
                    elif z[15]=='M' and prediction==1:
                        dme.append(1)
                    elif z[15]=='R' and prediction==2:
                        drusen.append(1)
                    elif z[15]=='O' and prediction==3:
                        normal.append(1)
                    elif z[15]=='N' and prediction!=0:
                        cnv.append(0)
                    elif z[15] == 'M' and prediction != 1:
                        dme.append(0)
                    elif z[15] == 'R' and prediction != 2:
                        drusen.append(0)
                    elif z[15] == 'O' and prediction != 3:
                        normal.append(0)
print(cnv)
print(dme)
print(drusen)
print(normal)
print('정확도cnv',np.mean(cnv))
print('정확도dme',np.mean(dme))
print('정확도drusen',np.mean(drusen))
print('정확도normal',np.mean(normal))
l=(np.mean(cnv)+np.mean(dme)+np.mean(drusen)+np.mean(normal)) / 4
print('전체',l)


