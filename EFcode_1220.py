import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.layers import Add, Activation, Dropout, Dense, Conv2D, BatchNormalization, MaxPooling2D, Concatenate, Model, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd


testECG = np.load('ValidationECGs.npy')  # The file that contains the ECG data


def multi_conv2D(x,num_kernel,activation="relu"):  # Defines a 2D convolutional module based on ResidualNet

    kreg = None  #regularizers.l2(0.01)
    sk = Conv2D(int(num_kernel*3), 1,activation=None, padding="same", kernel_regularizer=kreg)(x)
    sk = BatchNormalization()(sk)
    a = Conv2D(int(num_kernel), 1,activation=activation, padding="same", kernel_regularizer=kreg)(x)
    a = BatchNormalization()(a)
    a = Conv2D(num_kernel, 3, activation=activation, padding="same", kernel_regularizer=kreg)(a)
    a = BatchNormalization()(a)
    b = Conv2D(int(num_kernel), 1, activation=activation, padding="same", kernel_regularizer=kreg)(x)
    b = BatchNormalization()(b)
    b = Conv2D(int(num_kernel) ,3, activation=activation, padding="same", kernel_regularizer=kreg)(b)
    b = BatchNormalization()(b)
    b = Conv2D(num_kernel, 3, activation=activation, padding="same", kernel_regularizer=kreg)(b)
    b = BatchNormalization()(b)
    c = Conv2D(int(num_kernel), 1, activation=activation, padding="same", kernel_regularizer=kreg)(x)
    c = BatchNormalization()(c)
    res = Concatenate()([a,b,c])
    res = Add()([res,sk])
    res = BatchNormalization()(res)
    return res

SHAPE = (2500, 12, 1)  ## The input shape of the ECG data

def get_models_2D():  # Returns the model structure
    input1 = Input(SHAPE)
    initial_kernel_num = 64
    x = input1
    x = Conv2D(initial_kernel_num, (7,3),strides=(2,1), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = multi_conv2D(x, initial_kernel_num)
    x = multi_conv2D(x, initial_kernel_num)
    x = MaxPooling2D(pool_size=(3,1))(x)
    x = multi_conv2D(x, int(initial_kernel_num*1.5))
    x = multi_conv2D(x, int(initial_kernel_num*1.5))
    x = MaxPooling2D(pool_size=(3,1))(x)
    x = multi_conv2D(x, int(initial_kernel_num*2))
    x = multi_conv2D(x, int(initial_kernel_num*2))
    x = MaxPooling2D(pool_size=(2,1))(x)
    x = multi_conv2D(x, initial_kernel_num*3)
    x = multi_conv2D(x, initial_kernel_num*3)
    x = multi_conv2D(x, initial_kernel_num*4)
    x = MaxPooling2D(pool_size=(2,1))(x)
    x = multi_conv2D(x, initial_kernel_num*5)
    x = multi_conv2D(x, initial_kernel_num*6)
    x = multi_conv2D(x, initial_kernel_num*7)
    x = MaxPooling2D(pool_size=(2,1))(x)
    x = multi_conv2D(x, initial_kernel_num*8)
    x = multi_conv2D(x, initial_kernel_num*8)
    x = multi_conv2D(x, initial_kernel_num*8)
    x = MaxPooling2D(pool_size=(2,1))(x)
    x = multi_conv2D(x, initial_kernel_num*12)
    x = multi_conv2D(x, initial_kernel_num*14)
    x = multi_conv2D(x, initial_kernel_num*16)
    x = multi_conv2D(x, initial_kernel_num*16)
    x = multi_conv2D(x, initial_kernel_num*16)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=input1, outputs=x)
    return model


model = get_models_2D()
model.load_weights('weights.hdf5')
y_pred_val = model_best.predict(testECG, batch_size=30, verbose=1)
print(y_pred_val)



