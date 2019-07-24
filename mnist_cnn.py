# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:37:13 2019

@author: Himanshu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras as k
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense

Mnist =k.datasets.mnist
(X_train,Y_train),(X_test,Y_test) = Mnist.load_data(path = 'Mnist.npz')
X_train,X_test = X_train/255.0, X_test/255.0
img_rows , img_cols = 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
#num_category=10
#Y_train = k.utils.to_categorical(Y_train, num_category)
#Y_test = k.utils.to_categorical(Y_test, num_category)
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(input_shape), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(output_dim=100, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#model.summary()
model.fit(X_train,Y_train,nb_epoch = 10,validation_data=(X_test,Y_test))







