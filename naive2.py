"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import numpy as np
import resnet
import dataLoad
import dataLoad2

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet__nn.csv')
batch_size = 32
nb_classes = 5
nb_epoch = 100
data_augmentation = True

# input image dimensions
# img_rows, img_cols = 32, 32
img_rows, img_cols = 256, 256
# The CIFAR10 images are RGB.
# img_channels = 3
img_channels = 3

# The data, shuffled and split between train and test sets:
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train,y_train,_= dataLoad2.dataload("d:/data/permit/train",img_rows,img_cols,gray=0)
X_test,y_test,_= dataLoad2.dataload("d:/data/permit/test",img_rows,img_cols,gray=0)
# X_testr, _, testDict = dataLoad2.dataload("d:/data/permit/test", img_rows, img_cols, gray=0)

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.
mean_image.dump(open('mean1.npy','wb'))
model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
# model = resnet.ResnetBuilder.build_resnet_34((img_channels, img_rows, img_cols), nb_classes)
# model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, csv_logger])

else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    from keras.models import load_model
    model.load_weights('perimt1.h5')

    # saver = model.save_weights('my_model_weights_e120_res34_128_128.h5')

    # # Fit the model on the batches generated by datagen.flow().
    # model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
    #                     steps_per_epoch=X_train.shape[0] // batch_size,
    #                     validation_data=(X_test, Y_test),
    #                     epochs=nb_epoch, verbose=1, max_q_size=100,
    #                     callbacks=[lr_reducer, early_stopper, csv_logger])
    # model.save_weights('perimt1.h5')
    X_testr,_,testDict = dataLoad2.dataload("d:/data/permit/test",img_rows,img_cols,gray=0)
    import cv2

    X_testr = X_testr.astype('float32')
    X_testre =  X_testr.astype('int32')
    # subtract mean and normalize
    X_testr -= mean_image
    X_testr /= 128.



    classes = model.predict(X_testr, batch_size=128)
    output=dict()
    import csv


    # 写入数据

    # 写入的内容都是以列表的形式传入函数
    for index,value in enumerate(classes):
        # cv2.imwrite('C:/tempProjects/keras-resnet/vali7/'+str(np.argmax(value)+1)+'_'+testDict[index]+'.jpg', X_testre[index])
        cv2.imwrite('d:/git/keras-resnet/vali9/'+str(testDict[np.argmax(value)+1])+'.jpg', X_testre[index])

    print("write over")

    print()