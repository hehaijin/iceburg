import numpy as np
import keras
import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
import os.path as path
from keras.applications.vgg16 import VGG16

datapath='.\data\processed'

trainfile=open(path.join(datapath, 'train.json'))
data=pd.read_json(trainfile)

traindata= np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
traindata = traindata.reshape(1604,75,75,1)
traindata=np.concatenate((traindata,traindata,traindata),axis=3)
trainlabel=np.array(data["is_iceberg"])
trainlabel=np_utils.to_categorical(trainlabel,2)


testfile=open(path.join(datapath, 'test.json'))
data=pd.read_json(testfile)
testid=np.array(data["id"])  #8424

testdata= np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
testdata = testdata.reshape(8424,75,75,1)
testdata=np.concatenate((testdata,testdata,testdata),axis=3)


model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
#model_vgg16_conv.summary()



input = Input(shape=(75,75,3),name = 'band1')
output_vgg16_conv = model_vgg16_conv(input)

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = Dense(2, activation='softmax', name='predictions')(x)

my_model = Model(input=input, output=x)
my_model.summary()



my_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

my_model.fit(traindata, trainlabel, 
          batch_size=32, epochs=10, verbose=1)
result= my_model.predict(testdata,batch_size=32, verbose=1)


finalresult=np.zeros(8424)
for i in range(result.shape[0]):
	if result[i][1]>  result[i][0]:
		finalresult[i]=1

submission=pd.DataFrame({"id": testid, "is_iceberg": finalresult},index_col='id') 
submission.to_csv("submission.csv")



