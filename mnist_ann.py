# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:35:30 2019

@author: Himanshu
"""
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import keras as k

Mnist =k.datasets.mnist

(X_train,Y_train),(X_test,Y_test) = Mnist.load_data(path = 'Mnist.npz')

X_train,X_test = X_train/255.0, X_test/255.0

from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(k.layers.Flatten(input_shape=(28,28)))

model.add(Dense(output_dim = 128,init='uniform',activation='relu',input_dim =784))

model.add(Dense(output_dim = 128,init='uniform',activation='relu'))

model.add(Dense(output_dim = 10,init='uniform',activation='softmax'))

model.compile(optimizer = 'adam',loss ='sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit(X_train,Y_train,nb_epoch = 10)

Y_pred = model.predict(X_test)

loss_val,val_acc = model.evaluate(X_test,Y_test)

print(loss_val,val_acc)

'''
plt.imshow(X_test[1],cmap.)
print(np.argmax(Y_pred[1]))'''
