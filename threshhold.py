import pandas as pd
import os.path as path
import os


filename='submission.csv'
#outputfilename='submission2.csv'

datapath=path.join(os.getcwd(),filename)

submitfile=open(datapath)

df=pd.read_csv(submitfile)

for i, row in df.iterrows():
	if row["is_iceberg"] >= 0.9:
		df.set_value(i,'is_iceberg',0.9)
	if row["is_iceberg"] < 0.1:
		df.set_value(i,'is_iceberg',0.1)
		
df.to_csv("submission2.csv",index=False)
		


