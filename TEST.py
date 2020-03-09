import pickle
import os.path
import random
import numpy as np
import cv2
from keras.models import load_model, Model
import matplotlib.pyplot as plt

IMG_SIZE = 400
DIR = os.path.split(os.path.abspath(__file__))[0]
MODELDIR = os.path.join(DIR, 'model')
modelpath = MODELDIR+"/model400V2"
TDATADIR = os.path.join(DIR, 'trainingdata')
testpath = TDATADIR+"/test400"
drzewa = ["dab", "sosna"]




def loadtrainingdata(path):
    testdata = pickle.load(open(path, "rb"))
    return testdata

def predict(testdata, model):
    goodpredict = 0
    sosna = 0
    dab = 0
    dabgood = 0
    sosnagood = 0
    for img, number in testdata:
        print("To jest: "+drzewa[number])+" a wyszlo:"
        #plt.imshow(img,cmap="gray")
        #plt.show()
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        prediction = model.predict(img)
        result = drzewa[int(prediction[0][0])]
        print int(prediction[0][0])
        print(result)
        if (result == drzewa[number]):
            goodpredict= goodpredict+1
            if (number == 0):
                dabgood = dabgood + 1
            if (number == 1):
                sosnagood = sosnagood + 1
        if (number==0):
            dab = dab+1
        else:
            sosna = sosna+1

    print("Ilosc sosen:")
    print sosna
    print("Ilosc sosen dobrze:")
    print sosnagood
    print("Ilosc debow:")
    print dab
    print("Ilosc debow dobrze:")
    print dabgood
    print("Skutecznosc sosen:")
    print(100 * sosnagood / sosna)
    print("Skutecznosc debow:")
    print(100 * dabgood / dab)
    print("Procentowa skutecznoc:")
    print(100*goodpredict/len(testdata))







model= load_model(modelpath)
testdata = loadtrainingdata(testpath)
predict(testdata, model)


