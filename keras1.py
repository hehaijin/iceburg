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
import os
from keras.applications.vgg16 import VGG16
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma, denoise_tv_bregman, denoise_nl_means)
from skimage.filters import gaussian

import cv2

from keras.preprocessing.image import ImageDataGenerator





def create_dataset():
#read data
#datapath='./data/processed'
	datapath=path.join(os.getcwd(),"data","processed")

	trainfile=open(path.join(datapath, 'train.json'))
	data=pd.read_json(trainfile)

	traindata= np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
	traindata2= np.asarray([np.asarray(p).reshape(75,75) for p in data['band_2']])

	traindata = traindata.reshape(1604,75,75,1)
	traindata2 = traindata2.reshape(1604,75,75,1)


	traindata=np.concatenate((traindata,traindata2,(traindata+traindata2)/2),axis=3)
	traindata=np.array(traindata) 
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
	return traindata,trainlabel,testdata,testid

def preprocessing(data):
	datav = []
	datah = []
	datab = []
	datanew = []
	#can add more preprocessing here.
	#but do not overlap with the imagedatagenerator.
	for i in range(data.shape[0]):
		# m1=data[:,:,:,i].min()
		# m2=data[:,:,:,i].max()
		# if m2-m1 !=0:
		# 	data[:,:,:,i]=(data[:,:,:,i]-m1)/(m2-m1)
		m1=data[i,:,:,0]
		m2=data[i,:,:,1]
		m3=data[i,:,:,2]

		m1v=cv2.flip(m1,1)
		m1h=cv2.flip(m1,0)
		m1b=cv2.flip(m1,-1) # both v and h

		m2v=cv2.flip(m2,1)
		m2h=cv2.flip(m2,0)
		m2b=cv2.flip(m2,-1)

		m3v=cv2.flip(m3,1)
		m3h=cv2.flip(m3,0)
		m3b=cv2.flip(m3,-1)

		datav.append(np.dstack((m1v,m2v,m3v)))
		datah.append(np.dstack((m1h,m2h,m3h)))
		datab.append(np.dstack((m1b,m2b,m3b)))
	datav = np.array(datav)
	datah = np.array(datah)
	datab = np.array(datab)

	datanew = np.concatenate((data,datav,datah,datab))

	return datanew




def getModel():
#define model. it's simple VGG-16 with reduced full connectted layer.

	model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
	#model_vgg16_conv.summary()



	input = Input(shape=(75,75,3),name = 'band1')
	output_vgg16_conv = model_vgg16_conv(input)

	x = Flatten(name='flatten')(output_vgg16_conv)
	x = Dense(1024, activation='relu', name='fc1')(x)
	x = Dropout(0.8)(x)
	x = Dense(1024, activation='relu', name='fc2')(x)
	x = Dense(2, activation='softmax', name='predictions')(x)
	my_model = Model(input=input, output=x)
	my_model.summary()
	epochs=200
	learning_rate=0.0005
	decay_rate=learning_rate/(1+epochs/20)
	sgd = optimizers.SGD(lr=learning_rate,momentum=0.7,decay=decay_rate)

	my_model.compile(loss='binary_crossentropy',
				  optimizer=sgd,
				  metrics=['accuracy'])
	return my_model
	


def main():
	my_model=getModel()
	traindata,trainlabel,testdata,testid=create_dataset()
	
	#preprocessing 
	traindata=preprocessing(traindata) 
	trainlabel=np.concatenate((trainlabel,trainlabel,trainlabel,trainlabel))
	
	# #I do not know how to use imagedatagenerator
	# train_datagen = ImageDataGenerator(
	# 	#samplewise_center=True,
	# 	#samplewise_std_normalization=True,
	# 	#rotation_range=20,
	# 	#zoom_range=[0,0.3],
	# 	#width_shift_range=0.1,
	# 	#height_shift_range=0.1,
	# 	horizontal_flip=True,
	# 	vertical_flip=True)
	# test_datagen=ImageDataGenerator(
	# 	horizontal_flip=False,
	# 	vertical_flip=False)
        
    
	# my_model.fit_generator(train_datagen.flow(traindata,trainlabel,batch_size=32,shuffle=True),steps_per_epoch=50,epochs=120)
	
    #fit   
	my_model.fit(traindata,trainlabel,batch_size=32, epochs=30, verbose=1)
	result= my_model.predict(testdata,batch_size=32, verbose=1)
	
	#result2=my_model.predict_generator(test_datagen.flow(testdata,batch_size=1),verbose=1,steps=8424)
	#write to csv file.
	submission=pd.DataFrame({"id": testid, "is_iceberg": result[:,1]}) 
	submission.to_csv("submission.csv",index=False)
	#submission2=pd.DataFrame({"id": testid, "is_iceberg": result2[:,1]}) 
	#submission2.to_csv("submission2.csv",index=False)

if __name__ == "__main__":
    main()

