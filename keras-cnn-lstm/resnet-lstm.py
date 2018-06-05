import resnet
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM
img_rows, img_cols = 128, 128
# The CIFAR10 images are RGB.
# img_channels = 3
img_channels = 3
resnet = resnet.ResnetBuilder.build_resnet_50((img_rows,img_cols,3),img_channels)

resnet_lstm = LSTM(256)(resnet)
# resnet.add(LSTM(None,256))

print(resnet_lstm.summary())
# lstm = resnet.ResnetBuilder.build_resnet_lstm((img_rows,img_cols,3),img_channels)

# print(resnet.summary())