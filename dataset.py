from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os.path as path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt



def traindatagenerator(batchsize):
	datapath='.\data\processed'
	with open(path.join(datapath, 'train.json')) as trainfile:
		data=pd.read_json(trainfile)
		print(len(data.iloc[0]["band_1"]))
		gin = np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
		gin = gin.reshape(1604,75,75,1)
		datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False,rotation_range=90)
		datagen.fit(gin)
		return datagen.flow(gin, data['is_iceberg'], batch_size=batchsize)
			
def testdatagenerator(batchsize):
	datapath='.\data\processed'
	with open(path.join(datapath, 'test.json')) as testfile:
		data=pd.read_json(testfile)
		print(len(data.iloc[0]["band_1"]))
		gin = np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
		gin = gin.reshape(1604,75,75,1)
		return gin
	      
       
