import os
from os.path import join
import cv2
import numpy as np
from collections import defaultdict


def dictload(img_dirpath = 'e:/pic/train'):
    labelDict = dict()
    label=0
    for fileDirName in os.listdir(img_dirpath):
        labelDict[fileDirName] = label
        label=label+1
    return labelDict
def testdictload(dirpath = 'data/train.txt'):
    f = open(dirpath, "r")
    picDict = list()


    while True:
        line = f.readline()
        if line:
            pass  # do something here
            # name = line.split(' ')[0]
            # key,ext = os.path.splitext(name)
            picDict.append(line.strip())
        else:
            break
    f.close()

    return picDict

def dataload(img_dirpath= "e:/pic/train",img_w=300,img_h=300,val_ratio = 0.95,gray=0):
    labelDict= dictload()

    X_train = []
    y_train=[]
    X_val=[]
    y_val=[]
    for fileDirName in os.listdir(img_dirpath):
        dirpath = img_dirpath+"/"+fileDirName
        for filename in os.listdir(dirpath):
            name, ext = os.path.splitext(filename)
            if ext in ['.jpg']:
                img_filepath = join(dirpath, filename)
                img = cv2.imread(img_filepath)
                if gray==1:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, (img_w, img_h))
                img = img.astype(np.float32)
                X_train.append(img)
                y_train.append(labelDict[fileDirName])

    X_train,y_train,X_val,y_val = np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.int32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.int32)
    y_val = np.reshape(y_val,(len(y_val),1))
    y_train = np.reshape(y_train,(len(y_train),1))
    return X_train,y_train,labelDict


def testLoad(img_w=300,img_h=300,val_ratio = 0.95):
    # load y dict
    # picDict= testdictload("c:/tempProjects/keras-resnet/data/test.txt")
    picDict = list()

    # img_dirpath = "c:/tempProjects/keras-resnet/data/test"
    img_dirpath = "d:/git/keras-resnet/data/test"
    # X=[]
    # y=[]
    X_test = []

    for filename in os.listdir(img_dirpath):
        img_filepath = join(img_dirpath, filename)
        img = cv2.imread(img_filepath)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_w, img_h))
        img = img.astype(np.float32)
            # img /= 255
            # X.append(img)
            # y.append(labelDict[name])
        X_test.append(img)
        picDict.append(filename)
    X_test = np.asarray(X_test)
    X_test = X_test.astype(np.float32)
    return X_test,picDict
# X_train,y_train,X_val,y_val=dataload(50,50)
# print()