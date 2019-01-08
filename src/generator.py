# coding=utf-8
import skimage.transform
from keras.utils import Sequence
import numpy as np
#import cv2
import os
from scipy import misc
from skimage.transform import  resize
from init import *







def rgbf2bgr(rgbf):
	t = rgbf*255.0
	t = np.clip(t, 0.,255.0)
	bgr = t.astype(np.uint8)[..., ::-1]
	return bgr

def rgbf2rgb(rgbf):
	t = rgbf*255.0
	t = np.clip(t, 0.,255.0)
	rgb = t.astype(np.uint8)
	return rgb
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def read_image(file_path):
    Img=misc.imread(file_path)
    Img=rgb2gray(Img)/255.0
    return Img




class ImageSequence(Sequence):
    def __init__(self,  batch_size=4, input_size=(512, 512)):
        self.image_seq_path=DATA_PATH
        self.input_shape=input_size
        self.batch_size = batch_size
        self.epoch = 0
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]
        self.frames=sorted(os.listdir(self.image_seq_path+"ground_truth/"))

    def __len__(self):
        return (100)


    def read_image(self,file_path):
        Img=misc.imread(file_path)
	Img=rgb2gray(Img)/255.0
        return Img

    def __getitem__(self, idx):
        x_batch = []
        c=0
        num_frames=len(self.frames)*9/10
	end_idx=len(self.frames)
	start_idx=num_frames
        while(c<self.batch_size):
                              
            #s_idx =np.random.randint(0,num_frames)  #sence IDX
	    s_idx=np.random.randint(start_idx,end_idx)
                              
            I2_file=self.image_seq_path+"ground_truth/"+self.frames[s_idx] 
	    t1=np.random.randint(1,15)
 
            I1_file=self.image_seq_path+"rainy_image/"+self.frames[s_idx][:-4]+"_"+str(t1)+self.frames[s_idx][-4:]
            I1=read_image(I1_file)
            I2=read_image(I2_file)
            #I1=resize(I1,(self.SHAPE_Y,self.SHAPE_X,3))                
            #I2=resize(I2,(self.SHAPE_Y,self.SHAPE_X,3))                
            I1=resize(I1,(self.SHAPE_Y,self.SHAPE_X,1))                
            I2=resize(I2,(self.SHAPE_Y,self.SHAPE_X,1))                
	    #I1=I1[:,:,np.newaxis]
	    #I2=I2[:,:,np.newaxis]
            x_batch.append([I1,I2])
            c=c+1


        x_batch = np.array(x_batch, np.float32)
        #x_batch=np.zeros((4,2,240, 360, 1))
        #return ([x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]],y_batch)
        #return (x_batch[:,0,:,:,:],x_batch[:,0,:,:,:]-x_batch[:,1,:,:,:])
        return (x_batch[:,0,:,:,:],x_batch[:,1,:,:,:])
        #return x_batch

    def on_epoch_end(self):
        self.epoch += 1


    def get_test_data(self):
	end_idx=len(self.frames)
	start_idx=end_idx*9/10
	x_batch=[]
	#for s_idx in range(start_idx,start_idx+5,1):
	for s_idx in range(start_idx,end_idx,1):
            I2_file=self.image_seq_path+"ground_truth/"+self.frames[s_idx]
	    for j in range(1,15,1):
	            I1_file=self.image_seq_path+"rainy_image/"+self.frames[s_idx][:-4]+"_"+str(j)+self.frames[s_idx][-4:]
	            I1=read_image(I1_file)
	            I2=read_image(I2_file)
	            I1=resize(I1,(self.SHAPE_Y,self.SHAPE_X,1))
	            I2=resize(I2,(self.SHAPE_Y,self.SHAPE_X,1))
	            x_batch.append([I1,I2])
       	x_batch = np.array(x_batch, np.float32)
        return (x_batch[:,0,:,:,:],x_batch[:,1,:,:,:])



class TestImageSequence(Sequence):
    def __init__(self,  batch_size=4, input_size=(512, 512)):
        self.image_seq_path=DATA_PATH
        self.input_shape=input_size
        self.batch_size = batch_size
        self.epoch = 0
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]
        self.frames=sorted(os.listdir(self.image_seq_path+"ground_truth/"))

    def __len__(self):
        return (100)



    def __getitem__(self, idx):
	end_idx=len(self.frames)
	start_idx=end_idx*9/10
	x_batch=[]
	#for s_idx in range(start_idx,start_idx+1,1):
	for s_idx in range(start_idx,end_idx,1):
            I2_file=self.image_seq_path+"ground_truth/"+self.frames[s_idx]
	    for j in range(1,15,1):
	            I1_file=self.image_seq_path+"rainy_image/"+self.frames[s_idx][:-4]+"_"+str(j)+self.frames[s_idx][-4:]
	            I1=read_image(I1_file)
	            I2=read_image(I2_file)
	            I1=resize(I1,(self.SHAPE_Y,self.SHAPE_X,1))
	            I2=resize(I2,(self.SHAPE_Y,self.SHAPE_X,1))
	            x_batch.append([I1,I2])
       	x_batch = np.array(x_batch, np.float32)
        return (x_batch[:,0,:,:,:],x_batch[:,1,:,:,:])

    def on_epoch_end(self):
        self.epoch += 1




"""
from generator import *
A=ImageSequence()
t1,t2=A.__getitem__(3)
t1,t2=A.get_test_data()

"""





"""
from generator import *
A=ImageSequence()
t1,t2=A.__getitem__(3)
t1,t2=A.get_test_data()

"""



