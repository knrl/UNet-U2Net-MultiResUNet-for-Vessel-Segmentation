'''

    @author: mehmetkaanerol

'''
import random
import numpy as np
import tensorflow as tf
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Dropout, Activation

def conv2d_act(x, size, filters=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'):
    x = Conv2D(size, filters, padding=padding, dilation_rate=dilation_rate, kernel_initializer=kernel_initializer)(x)
    x = Activation(activation)(x)
    return x

def get_Unet(x):
    #Contraction path
    c1 = conv2d_act(x, 16)
    c1 = Dropout(0.1)(c1)
    c1 = conv2d_act(c1, 16)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv2d_act(p1, 32)
    c2 = Dropout(0.1)(c2)
    c2 = conv2d_act(c2, 32)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = conv2d_act(p2, 64)
    c3 = Dropout(0.2)(c3)
    c3 = conv2d_act(c3, 64)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv2d_act(p3, 128) 
    c4 = Dropout(0.2)(c4)
    c4 = conv2d_act(c4, 128)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = conv2d_act(p4, 256) 
    c5 = Dropout(0.3)(c5)
    c5 = conv2d_act(c5, 256) 

    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_act(u6, 128) 
    c6 = Dropout(0.2)(c6)
    c6 = conv2d_act(c6, 128) 
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_act(u7, 64) 
    c7 = Dropout(0.2)(c7)
    c7 = conv2d_act(c7, 64) 
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_act(u8, 32) 
    c8 = Dropout(0.1)(c8)
    c8 = conv2d_act(u8, 32) 
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_act(u9, 16) 
    c9 = Dropout(0.1)(c9)
    c9 = conv2d_act(c9, 16) 
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return keras.Model(inputs=[inputs], outputs=[outputs])
