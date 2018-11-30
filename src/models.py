import os,sys
from keras.engine.topology import Layer
import numpy as np
from keras.layers import initializers,constraints
import tensorflow as tf
from keras.models import Sequential,Model
from keras.utils import conv_utils
from keras import backend as K
import numpy as np
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D,Concatenate
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras_contrib.losses import DSSIMObjective
from keras.constraints import *
#from morph_dense import *
from sklearn.datasets import  make_circles
from sklearn import datasets
from keras.layers import Activation, Dense,Dropout
import pandas
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
from morph_layers2D import *
from matplotlib.colors import ListedColormap
import keras
from generator import *
from keras.layers import maximum
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk,rectangle
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Add, Dropout, concatenate
import time








def model_morphoN_small():
    I=Input(shape=(512,512,1))
    #I=Input(shape=(None,None,1))
    z1=Dilation2D(1, (8,8),padding="same",strides=(1,1))(I)
    z2=Erosion2D(1, (8,8),padding="same",strides=(1,1))(I)
    for i in range(2):
        for j in range(2):
                z1=Dilation2D(1, (8,8),padding="same",strides=(1,1))(z1)
                z2=Erosion2D(1, (8,8),padding="same",strides=(1,1))(z2)
        for j in range(2):
                z1=Erosion2D(1, (8,8),padding="same",strides=(1,1))(z1)
                z2=Dilation2D(1, (8,8),padding="same",strides=(1,1))(z2)

    z1=Erosion2D(1, (8,8),padding="same",strides=(1,1))(z1)
    w1=Conv2D(2,(8,8),activation="tanh",padding="same")(z1)
    w1=Conv2D(3,(8,8),activation="tanh",padding="same")(w1)
    w1=Conv2D(1,(8,8),activation="sigmoid",padding="same")(w1)
    z2=Dilation2D(1, (8,8),padding="same",strides=(1,1))(z2)
    w2=Conv2D(2,(8,8),activation="tanh",padding="same")(z2)
    w2=Conv2D(3,(8,8),activation="tanh",padding="same")(w2)
    w2=Conv2D(1,(8,8),activation="sigmoid",padding="same")(w2)

    z3=CombDense_new(units=2)([z1,z2,w1,w2])
    model=Model(inputs=[I],outputs=[z3])
    return model




def model_morphoN():
    I=Input(shape=(512,512,1))
    z1=I
    z2=I
    for i in range(2):
        for j in range(2):
                z1=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z1)
                z2=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z2)
        for j in range(2):
                z1=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z1)
                z2=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z2)

    z1=Erosion2D(1, (8,8),padding="same",strides=(1,1))(z1)
    w1=Conv2D(2,(8,8),activation="tanh",padding="same")(z1)
    w1=Conv2D(3,(8,8),activation="tanh",padding="same")(w1)
    w1=Conv2D(1,(8,8),activation="sigmoid",padding="same")(w1)
    z2=Dilation2D(1, (8,8),padding="same",strides=(1,1))(z2)
    w2=Conv2D(2,(8,8),activation="tanh",padding="same")(z2)
    w2=Conv2D(3,(8,8),activation="tanh",padding="same")(w2)
    w2=Conv2D(1,(8,8),activation="sigmoid",padding="same")(w2)

    z3=CombDense_new(units=2)([z1,z2,w1,w2])
    model=Model(inputs=[I],outputs=[z3])
    return model



def model_path1():
    I=Input(shape=(512,512,1))
    #I=Input(shape=(None,None,1))
    #z1=Dilation2D(4, (8,8),padding="same",strides=(1,1))(I)
    z1=I
    for i in range(2):
        for j in range(2):
                z1=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z1)
        for j in range(2):
                z1=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z1)

    z1=Erosion2D(1, (8,8),padding="same",strides=(1,1))(z1)    
    model=Model(inputs=[I],outputs=[z1])
    return model

def model_path2():
    I=Input(shape=(512,512,1))


    #z2=Erosion2D(4, (8,8),padding="same",strides=(1,1))(I)
    z2=I
    for i in range(2):
        for j in range(2):
                z2=Erosion2D(4, (8,8),padding="same",strides=(1,1))(z2)
        for j in range(2):
                z2=Dilation2D(4, (8,8),padding="same",strides=(1,1))(z2)

    z2=Dilation2D(1, (8,8),padding="same",strides=(1,1))(z2)
    model=Model(inputs=[I],outputs=[z2])
    return model


def unet_down_1(filter_count, inputs, activation='relu', pool=(2, 2), n_layers=3):
    down = inputs
    for i in range(n_layers):
        down = Conv2D(filter_count, (3, 3), padding='same', activation=activation)(down)
        down = BatchNormalization()(down)

    if pool is not None:
        x = MaxPooling2D(pool, strides=pool)(down)
    else:
        x = down
    return (x, down)

def unet_up_1(filter_count, inputs, down_link, activation='relu', n_layers=3):
    reduced = Conv2D(filter_count, (1, 1), padding='same', activation=activation)(inputs)
    up = UpSampling2D((2, 2))(reduced)
    up = BatchNormalization()(up)
    link = Conv2D(filter_count, (1, 1), padding='same', activation=activation)(down_link)
    link = BatchNormalization()(link)
    up = Add()([up,link])
    for i in range(n_layers):
        up = Conv2D(filter_count, (3, 3), padding='same', activation=activation)(up)
        up = BatchNormalization()(up)
    return up



def model_CNN(input_shape=(512,512,1)):
    n_layers_down = [2, 3, 3, 3, 3, 3]
    n_layers_up = [2, 3, 3, 3, 3, 3]
    n_filters_down = [16,32,64, 96, 144, 192]
    n_filters_up = [16,32,64, 96, 144, 192]
    n_filters_center=256
    n_layers_center=4
    activation='relu'
    inputs = Input(shape=input_shape)
    x = inputs
    x = BatchNormalization()(x)
    xbn = x
    depth = 0
    back_links = []
    for n_filters in n_filters_down:
        n_layers = n_layers_down[depth]
        x, down_link = unet_down_1(n_filters, x, activation=activation, n_layers=n_layers)
        back_links.append(down_link)
        depth += 1

    center, _ = unet_down_1(n_filters_center, x, activation=activation, pool=None, n_layers=n_layers_center)


    # center
    x1 = center
    while depth > 0:
        depth -= 1
        link = back_links.pop()
        n_filters = n_filters_up[depth]
        n_layers = n_layers_up[depth]
        x1 = unet_up_1(n_filters, x1, link, activation=activation, n_layers=n_layers)
        if depth <= 1:
            x1 = Dropout(0.25)(x1)

    x1 = concatenate([x1,xbn])
    x1 = Conv2D(16, (3, 3), padding='same', activation=activation)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(16, (3, 3), padding='same', activation=activation)(x1)
    x1 = BatchNormalization()(x1)

    x1 = Conv2D(1, (1, 1), activation='sigmoid')(x1)
    model = Model(inputs=inputs, outputs=x1)
    return model







