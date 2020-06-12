import numpy as np
from PIL import Image
import glob
test='/home/kim/kfold/5/test/'
a=[]
b=[]
c=[]
d=[]
categories=['CNV','DME','DRUSEN','NORMAL']
def visualize_predictions(model,real):
    for idx, categorie in enumerate(categories):
        files = glob.glob(test + categorie + "/*")
        for idx2, file in enumerate(files):
            img_path = Image.open(file)
            img_path = img_path.convert('RGB')
            img_path = img_path.resize((299, 299))
            data = np.asarray(img_path)
            data = data.reshape((1, 299, 299, 3))
            data = data / 255
            # result = model.predict(data)
            # prediction = real.predict_classes(result)
            prediction = real.predict(data)
            prediction = np.argmax(prediction)

            if prediction == 0 and file[24] == 'N':
                a.append(1)
            elif prediction == 1 and file[24] == 'M':
                b.append(1)
            elif prediction == 2 and file[24] == 'R':
                c.append(1)
            elif prediction == 3 and file[24] == 'O':
                d.append(1)
            elif file[24] == 'N' and prediction != 0:
                a.append(0)
            elif file[24] == 'M' and prediction != 1:
                b.append(0)
            elif file[24] == 'R' and prediction != 2:
                c.append(0)
            elif file[24] == 'O' and prediction != 3:
                d.append(0)
    print('CNV {} , DME {} , DRUSEN {} , NORMAL {}'.format(a,b,c,d))
    print('')
    print('CNV 평균 {}, DME 평균 {} , DRUSEN 평균 {} , NORMAL 평균 {}'.format(np.mean(a),np.mean(b),np.mean(c),np.mean(d)))
    l=(np.mean(a)+np.mean(b)+np.mean(c)+np.mean(d)) / 4
    print(l)
import keras

print('인셉션레스넷')
model = keras.models.load_model('./image_model/VGG19_model_allFine')
conv_base = keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
visualize_predictions(conv_base, model)  # 테스트할 이미지들의 갯수 설정가능


