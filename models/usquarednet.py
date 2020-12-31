import tensorflow as tf
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import UpSampling2D, Dropout, Softmax, BatchNormalization, Activation

def conv2d_bn_act(x, size, filters=(3,3), padding='same', dilation_rate=(1,1), activation='relu', kernel_initializer='he_normal'):
    x = Conv2D(size, filters, padding=padding, dilation_rate=dilation_rate, kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x
    
def get_en1(hx):
    c1i = conv2d_bn_act(hx, 16)
    c1 = conv2d_bn_act(c1i, 16)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv2d_bn_act(p1, 32)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv2d_bn_act(p2, 64)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv2d_bn_act(p3, 128)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    c5 = conv2d_bn_act(p4, 256)
    c6 = conv2d_bn_act(c5, 256, dilation_rate=(2,2))
    c7 = conv2d_bn_act(c6, 256)
    c7 = tf.keras.layers.Dropout(0.2)(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = conv2d_bn_act(u8, 128)
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = conv2d_bn_act(u9, 64)
    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = conv2d_bn_act(u10, 32)
    u11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c1])
    c11 = conv2d_bn_act(u11, 16)
    
    return c11 + c1i

def get_en2(hx):
    c2i = conv2d_bn_act(hx, 32)
    c2 = conv2d_bn_act(c2i, 32)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv2d_bn_act(p2, 64)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv2d_bn_act(p3, 128)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv2d_bn_act(p4, 256)
    c6 = conv2d_bn_act(c5, 256, dilation_rate=(2, 2))
    c7 = conv2d_bn_act(c6, 256)
    c7 = tf.keras.layers.Dropout(0.2)(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = conv2d_bn_act(u8, 128)
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = conv2d_bn_act(u9, 64)
    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = conv2d_bn_act(u10, 32)

    return c10 + c2i

def get_en3(hx):
    c3i = conv2d_bn_act(hx, 64)
    c3 = conv2d_bn_act(c3i, 64)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv2d_bn_act(p3, 128)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv2d_bn_act(p4, 256)
    c6 = conv2d_bn_act(c5, 256, dilation_rate=(2, 2))
    c7 = conv2d_bn_act(c6, 256)
    c7 = tf.keras.layers.Dropout(0.2)(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = conv2d_bn_act(u8, 128)
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = conv2d_bn_act(u9, 64)

    return c9 + c3i

def get_en4(hx):
    c4i = conv2d_bn_act(hx, 128)
    c4 = conv2d_bn_act(c4i, 256)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv2d_bn_act(p4, 256)
    c6 = conv2d_bn_act(c5, 256, dilation_rate=(2, 2))
    c7 = conv2d_bn_act(c6, 256)
    c7 = tf.keras.layers.Dropout(0.2)(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = conv2d_bn_act(u8, 128)
    
    return c8 + c4i

def get_en5(hx):
    c5i = conv2d_bn_act(hx, 256)
    c5 = conv2d_bn_act(c5i, 256)
    c6 = conv2d_bn_act(c5, 256, dilation_rate=(2, 2))
    c7 = conv2d_bn_act(c6, 256, dilation_rate=(4, 4))
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    
    c8 = conv2d_bn_act(c7, 256, dilation_rate=(8, 8))
    c9 = conv2d_bn_act(c8, 256, dilation_rate=(4, 4))
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = concatenate([c9, c7])
    c10 = conv2d_bn_act(c9, 256, dilation_rate=(2, 2))
    c10 = concatenate([c10, c6])
    c11 = conv2d_bn_act(c10, 256)

    return c11 + c5i

def get_U2net(x):
    hx1 = get_en1(x)
    hx = MaxPooling2D(pool_size=(2, 2))(hx1)
    hx2 = get_en2(hx)
    hx = MaxPooling2D(pool_size=(2, 2))(hx2)
    hx3 = get_en3(hx)
    hx = MaxPooling2D(pool_size=(2, 2))(hx3)
    hx4 = get_en4(hx)
    hx = MaxPooling2D(pool_size=(2, 2))(hx4)
    hx5 = get_en5(hx)
    hx5dup = UpSampling2D(size=(2, 2), interpolation='bilinear')(hx5)

    con4 = concatenate([hx5dup,hx4])
    hx4d = get_en4(con4)
    hx4dup = UpSampling2D(size=(2, 2), interpolation='bilinear')(hx4d)
    con3 = concatenate([hx4dup,hx3])
    hx3d = get_en3(con3)
    hx3dup = UpSampling2D(size=(2, 2), interpolation='bilinear')(hx3d)
    con2 = concatenate([hx3dup,hx2])
    hx2d = get_en2(con2)
    hx2dup = UpSampling2D(size=(2, 2), interpolation='bilinear')(hx2d)
    con1 = concatenate([hx2dup,hx1])
    hx1d = get_en1(con1)

    d1 = Conv2D(16, (3, 3), activation='sigmoid', padding='same')(hx1d)
    d2 = Conv2D(32, (3, 3), activation='sigmoid', padding='same')(hx2d)
    d2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)
    d3 = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(hx3d)
    d3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)
    d4 = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(hx4d)
    d4 = UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)
    d5 = Conv2D(256, (3, 3), activation='sigmoid', padding='same')(hx5)
    d5 = UpSampling2D(size=(16, 16), interpolation='bilinear')(d5)

    con_last = concatenate([d1,d2,d3,d4,d5])
    d0 = Conv2D(1, (1, 1), activation='sigmoid')(con_last)
    
    return tf.keras.Model(inputs=[inputs], outputs=[d0])