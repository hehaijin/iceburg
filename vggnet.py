from datetime import datetime
import math
import time
import tensorflow as tf
import numpy as np
import dataset


def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
	#input data through a conv2d with parameters to set kernel and stride, and append kernel to p
	#
	n_in=input_op.get_shape()[-1].value #the channel number
	with tf.name_scope(name) as scope:
		#here use get_variable
		kernel=tf.get_variable(scope+"w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
		conv=tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
		bias_init_val=tf.constant(0.0,shape=[n_out],dtype=tf.float32)
		#trainable flag adds to GraphKeys.TRAINABLE_VARIABLES. The various Optimizer classes use this collection as the default list of variables to optimize.
		biases=tf.Variable(bias_init_val,trainable=True,name='b')
		#bias_add is special case of tf.add, where 2 parameters can be of different type or shape. 
		z=tf.nn.bias_add(conv,biases)
		activation=tf.nn.relu(z,name=scope)
		p+=[kernel,biases]
		return activation
	
def fc_op(input_op,name,n_out,p):
	#this op returns a 1D tensor of shape [n_out]
	n_in=input_op.get_shape()[-1].value
	with tf.name_scope(name) as scope:
		kernel=tf.get_variable(scope+"w",shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
		biases=tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
		#relu_layer(x,w,b)=tf.nn.relu(x*weight+b)
		activation=tf.nn.relu_layer(input_op,kernel,biases,name=scope)
		p+=[kernel,biases]
		return activation
		
def mpool_op(input_op,name,kh,kw,dh,dw):
	#a thin wrapper on max_pool function.
	return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)
	

def inference_op(input_op,keep_prob):
	p=[]
	#conv1
	conv1_1=conv_op(input_op,'conv1_1',3,3,64,1,1,p)
	conv1_2=conv_op(conv1_1,'conv1_2',3,3,64,1,1,p)
	pool1=mpool_op(conv1_2,'pool1',2,2,2,2)
	#conv2 
	conv2_1=conv_op(pool1,'conv2_1',3,3,128,1,1,p)
	conv2_2=conv_op(conv2_1,'conv2_2',3,3,128,1,1,p)
	pool2=mpool_op(conv2_2,'pool2',2,2,2,2)
	#conv3 
	conv3_1=conv_op(pool2,'conv3_1',3,3,256,1,1,p)
	conv3_2=conv_op(conv3_1,'conv3_2',3,3,256,1,1,p)
	conv3_3=conv_op(conv3_2,'conv3_3',3,3,256,1,1,p)
	pool3=mpool_op(conv3_3,'pool3',2,2,2,2)
	#conv4
	conv4_1=conv_op(pool3,'conv4_1',3,3,512,1,1,p)
	conv4_2=conv_op(conv4_1,'conv4_2',3,3,512,1,1,p)
	conv4_3=conv_op(conv4_2,'conv4_3',3,3,512,1,1,p)
	pool4=mpool_op(conv4_3,'pool4',2,2,2,2)
	#conv5
	conv5_1=conv_op(pool4,'conv5_1',3,3,512,1,1,p)
	conv5_2=conv_op(conv5_1,'conv5_2',3,3,512,1,1,p)
	conv5_3=conv_op(conv5_2,'conv5_3',3,3,512,1,1,p)
	pool5=mpool_op(conv5_3,'pool5',2,2,2,2)
	#flatten
	shp=pool5.get_shape()
	#image tensor for convolution is 4D. first dimension is number of images.
	flattened_shape=shp[1].value*shp[2].value*shp[3].value
	reshp1=tf.reshape(pool5,[-1,flattened_shape],name='resh1')
	fc6=fc_op(reshp1,name='fc6',n_out=1024,p=p)
	fc6_drop=tf.nn.dropout(fc6,keep_prob,name="fc6_drop")
	fc7=fc_op(fc6_drop,name='fc7',n_out=1024,p=p)
	fc7_drop=tf.nn.dropout(fc7,keep_prob,name='fc_drop')
	fc8=fc_op(fc7_drop,name='fc8',n_out=2,p=p)
	softmax=tf.nn.softmax(fc8)
	predictions=tf.argmax(softmax,1)
	return predictions,softmax,fc8,p
	
	
def tensorflow_run(session,target,feed,info_string,num_batches):
	
	num_steps_burn_in=10
	total_duration=0.0
	total_duration_squared=0.0
	for i in range(num_batches+num_steps_burn_in):
		start_time=time.time()
		_=session.run(target,feed_dict=feed)

	
	
	
def run_benchmark(batch_size,num_batches):
	with tf.Graph().as_default():
		image_size=224
		images=tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))
		keep_prob=tf.placeholder(tf.float32)
		predictions,softmax,fc8,p=inference_op(images,keep_prob)
		init=tf.global_variables_initializer()
		sess=tf.Session()
		sess.run(init)
		time_tensorflow_run(sess,predictions,{keep_prob:1.0},"forward",num_batches)
		objective=tf.nn.l2_loss(fc8)
		grad=tf.gradients(objective,p)
		time_tensorflow_run(sess,grad,{keep_prob:0.5},"forward-backward",num_batches)
	

def runmodel(batch_size,num_batches):
	with tf.Graph().as_default():
		with tf.Session().as_default() as sess:
		#sess=tf.Session()
			
			images=tf.placeholder(tf.float32, [batch_size,75,75,1])
			predictions,softmax,a,b=inference_op(images,1)
			labels=tf.placeholder(tf.float32, [batch_size,2])
			#predictions2=tf.to_float(predictions, name='ToFloat')
			#print(type(labels))
			#print(type(softmax))
			cross_entropy= tf.reduce_mean(-tf.reduce_sum(labels * tf.log(softmax),reduction_indices=[1]))
			train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) 
			init=tf.global_variables_initializer()
			sess.run(init)
			dg=dataset.traindatagenerator(batch_size)
			stepcount=0
			for x,y in dg:
				y2=np.zeros((len(y),2))
				for i in range(len(y)):
					if y[i]==0:
						y2[i,0]=1
					else:
						y2[i,1]=1
				print(stepcount)
				train_step.run({images:x,labels:y2})
				stepcount=stepcount+1
				if stepcount > num_batches:
					break
			
		

batch_size=32 
num_batches=10
runmodel(16,16)
	
