#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 09:52:34 2019

@author: sleek_eagle
"""

import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
import numpy as np
np.random.seed(1000)
from os import listdir
from os.path import isfile, join
from random import randint
from keras import optimizers
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K

def get_cnn():
    inputs = Input(shape=(40,200,1))
    #1st cnn layer
    x = Conv2D(filters=16,kernel_size=(10,10),strides=(2,2))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    #2nd cnn layer
    x = Conv2D(filters=32,kernel_size=(10,10),strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    #flatten
    x=Flatten()(x)
    
    #1st FC layer
    x = Dense(716)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.6)(x)
    
    #2nd FC layer
    x = Dense(716)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)
    
    #softmax
    predictions = Dense(7, activation='softmax')(x)
    
    model=Model(inputs=inputs,outputs=predictions)
    sgd = optimizers.SGD(lr=0.001, momentum=0.6, decay=0.0, nesterov=False)
    rms=optimizers.RMSprop(lr=0.001, rho=0.9)
    lrs = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20,verbose=1)
    model.compile(optimizer=rms, loss='categorical_crossentropy',metrics=['accuracy'])
    return model