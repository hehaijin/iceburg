import cv2
import json
import pandas as pd
import os
import os.path as path
import numpy as np

def main():
	#datapath='.\data\processed'
	outputpath='pics'
	with open(path.join("data","processed", 'train.json')) as trainfile:
		data=pd.read_json(trainfile)
		gin = np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
		gin = gin.reshape(1604,75,75,1)
		gin2 = np.asarray([np.asarray(p).reshape(75,75) for p in data['band_1']])
		gin2 = gin2.reshape(1604,75,75,1)
		for i in range(1604):
			print(i)
			t=data.iloc[i]['is_iceberg']
			img= gin[i,:,:,0]
			img=(img-img.min())*255/(img.max()-img.min())
			img=img.astype("uint8")
			#print(img)
			#print(img.min())
			#print(img.max())
			
			img = cv2.GaussianBlur(img,(3,3),0)
			#ret3,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			#img=cv2.equalizeHist(img)
			#img=cv2.fastNlMeansDenoising(img)
			#print(img)
			img2=gin2[i,:,:,0]
			img2=(img2-img2.min())*255/(img2.max()-img2.min())
			img2=img2.astype("uint8")
			#print(img2)
			img2=cv2.fastNlMeansDenoising(img2)
			cv2.imwrite(path.join(outputpath,"band1-"+str(i)+"-"+str(t)+".jpg"),img)
			cv2.imwrite(path.join(outputpath,"band2-"+str(i)+"-"+str(t)+".jpg"),img2)
		
		


if __name__=="__main__":
	main()
