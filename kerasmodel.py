import numpy as np
from keras import optimizers
import keras
import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
import os.path as path
from keras.applications.vgg16 import VGG16



#read data
datapath='./data/processed'

trainfile=open(path.join(datapath, 'train.json'))
data=pd.read_json(trainfile)

traindata= np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
traindata2= np.asarray([np.asarray(p).reshape(75,75) for p in data['band_2']])

traindata = traindata.reshape(1604,75,75,1)
traindata2 = traindata2.reshape(1604,75,75,1)

traindata=np.concatenate((traindata,traindata2,(traindata+traindata2)/2),axis=3)
trainlabel=np.array(data["is_iceberg"])
trainlabel=np_utils.to_categorical(trainlabel,2)


testfile=open(path.join(datapath, 'test.json'))
data=pd.read_json(testfile)
testid=np.array(data["id"])  #8424

testdata= np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
testdata2= np.asarray([np.asarray(p).reshape(75,75) for p in data['band_2']])

testdata = testdata.reshape(8424,75,75,1)
testdata2 = testdata2.reshape(8424,75,75,1)

testdata=np.concatenate((testdata,testdata2,(testdata+testdata2)/2),axis=3)






#define model. it's simple VGG-16 with reduced full connectted layer.

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

sgd = optimizers.SGD(lr=0.0001)

my_model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])



#train model
my_model.fit(traindata, trainlabel, 
          batch_size=32, epochs=1000, verbose=1)
          
#predict          
result= my_model.predict(testdata,batch_size=32, verbose=1)


#write to csv file.
submission=pd.DataFrame({"id": testid, "is_iceberg": result[:,1]}) 
submission.to_csv("submission.csv",index=False)


