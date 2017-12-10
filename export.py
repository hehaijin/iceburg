import cv2
import json
import pandas as pd


datapath='.\data\processed'
with open(path.join(datapath, 'train.json')) as trainfile:
	data=pd.read_json(trainfile)
	gin = np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
	gin = gin.reshape(1604,75,75,1)
	gin2 = np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
	gin2 = gin2.reshape(1604,75,75,1)
	for i in range(1604):
		t=data.iloc[i]['is_iceberg']
		cv2.imwrite("band1-"+str(i)+t+".jpg",gin[i,:,:,:])
		cv2.imwrite("band2-"+str(i)+t+".jpg",gin[i,:,:,:])
