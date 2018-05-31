import os
from os.path import join
import cv2
import numpy as np


def dictload(dirpath = 'data/train.txt'):
    f = open(dirpath, "r")
    labelDict = dict()
    while True:
        line = f.readline()
        if line:
            pass  # do something here
            name = line.split(' ')[0]
            label = line.split(' ')[1].strip()
            key,ext = os.path.splitext(name)
            labelDict[key]=label
        else:
            break
    f.close()
    return labelDict

def dataload(img_w=300,img_h=300,val_ratio = 0.95):
    # load y dict
    labelDict = dictload("d:/git/keras-resnet/data/train.txt")


    img_dirpath = "d:/git/keras-resnet/data/train"
    X=[]
    y=[]

    for filename in os.listdir(img_dirpath):
        name, ext = os.path.splitext(filename)
        if ext in ['.jpg']:
            img_filepath = join(img_dirpath, filename)
            img = cv2.imread(img_filepath)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_w, img_h))
            img = img.astype(np.float32)
            # img /= 255
            X.append(img)
            y.append(labelDict[name])
    # X,y=np.asarray(X), np.asarray(y)
    # X =X.astype(np.float32)
    # y =y.astype(np.float32)
    #train validate

    trainLen = int(len(X)*0.95)
    valLen = len(X)-trainLen
    X_train=[]
    y_train=[]
    X_val=[]
    y_val=[]

    for index,value in enumerate(X):
        if index<trainLen:
            X_train.append(value)
            y_train.append(y[index])
        else:
            X_val.append(value)
            y_val.append(y[index])
    X_train,y_train,X_val,y_val = np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.uint32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.uint32)
    y_val = np.reshape(y_val,(1,len(y_val)))
    y_train = np.reshape(y_train,(1,len(y_train)))


    return X_train,y_train,X_val,y_val

X_train,y_train,X_val,y_val=dataload(50,50)
print()