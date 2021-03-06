import numpy as np
from keras import optimizers
import keras
import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
import os.path as path
import os
from keras.applications.vgg16 import VGG16
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma, denoise_tv_bregman, denoise_nl_means)
from skimage.filters import gaussian

#import cv2

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





def getModel():
#define model. it's simple VGG-16 with reduced full connectted layer.

	model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
	#model_vgg16_conv.summary()



	input = Input(shape=(75,75,3),name = 'band1')
	output_vgg16_conv = model_vgg16_conv(input)

	x = Flatten(name='flatten')(output_vgg16_conv)
	x = Dense(1024, activation='relu', name='fc1')(x)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation='relu', name='fc2')(x)
	x = Dropout(0.5)(x)
	x = Dense(2, activation='softmax', name='predictions')(x)
	
	my_model = Model(input=input, output=x)
	my_model.summary()
	epochs=200
	learning_rate=0.0002
	decay_rate=0.001
	sgd = optimizers.SGD(lr=learning_rate,momentum=0.8)

	my_model.compile(loss='binary_crossentropy',
				  optimizer=sgd,
				  metrics=['accuracy'])
	return my_model
	





def preprocessing(data):
	#can add more preprocessing here.
	#but do not overlap with the imagedatagenerator.
	print(data.shape)
	for i in range(3):
		for j in range(data.shape[0]):
			m1=data[j,:,:,i].min()
			m2=data[j,:,:,i].max()
			if m2-m1 !=0:
				data[j,:,:,i]=(data[j,:,:,i]-m1)/(m2-m1)
	return data




def main():
	my_model=getModel()
	traindata,trainlabel,testdata,testid=create_dataset()

	#preprocessing 
	#traindata=preprocessing(traindata) 
	#testdata=preprocessing(testdata)
	
	X_train, X_valid, y_train, y_valid = train_test_split(traindata, trainlabel, test_size=0.25)

	train_datagen = ImageDataGenerator(
		#samplewise_center=True,
		#samplewise_std_normalization=True,
		#rotation_range=20,
		#zoom_range=0.2,
		#width_shift_range=0.1,
		#height_shift_range=0.1,
		#horizontal_flip=True,
		vertical_flip=True)
	test_datagen=ImageDataGenerator(
		horizontal_flip=False,
		vertical_flip=False)
        
    
	my_model.fit_generator(train_datagen.flow(X_train,y_train,batch_size=32,shuffle=True),steps_per_epoch=50,epochs=400,validation_data=(X_valid,y_valid))
	
    #fit   
    #my_model.fit(traindata, trainlabel, 
    #      batch_size=32, epochs=100, verbose=1)          
	#predict          
	
	result= my_model.predict(testdata,batch_size=32, verbose=1)
	
	
	#result2=my_model.predict_generator(test_datagen.flow(testdata,batch_size=1),verbose=1,steps=8424)
	
	#write to csv file.
	submission=pd.DataFrame({"id": testid, "is_iceberg": result[:,1]}) 
	submission.to_csv("submission.csv",index=False)
	

if __name__ == "__main__":
    main()

