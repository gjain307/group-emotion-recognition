#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:42:29 2018

@author: shreya
"""
import keras
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from keras.regularizers import l1_l2, Regularizer
from keras.engine import Layer
from theano import function, shared, printing
from keras.engine import InputSpec
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation,Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import theano.tensor as T
import scipy.io as sio
seed = 7
from sklearn.utils import shuffle
from keras.utils import np_utils
#%%
from keras.utils.generic_utils import get_custom_objects
#
import h5py as h5
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

dataset=sio.loadmat('/data/shreya/VIP_MIL/vip_data.mat')
x=dataset['data'].astype('float32')
y=dataset['label']
x,y = shuffle(x,y, random_state=0)
y = np_utils.to_categorical(y)

batch_size = 20
faces_per_image=3;
feature_per_face=4096;

class SortLayer(Layer):
    # Rerank is difficult. It is equal to the number of points (>0.5) to be fixed number.
    def __init__(self,k=1,label=1,**kwargs):
        # k is the factor we force to be 1
        self.k = k*1.0
        self.label = label
        
        self.input_spec = [InputSpec(ndim=4)]
        
        super(SortLayer, self).__init__(**kwargs)

    def build(self,input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, x,mask=None):
        
        
      
        #btch_size= batch_size
#    
        x=K.reshape(x, (-1, faces_per_image))
        #T.tensor.as_tensor_variable(np.asarray(x))
        print(np.shape(x))
        
        first = K.max(x, axis=1)
        m1 = K.argmax(x, axis=1)
        m1 = K.one_hot(m1, 3)
        x = x * (1 - m1)
        
        second = K.max(x, axis=1)
        m1 = K.argmax(x, axis=1)
        m1 = K.one_hot(m1, 3)
        x = x * (1 - m1)
        
        third = K.max(x, axis=1)
        m1 = K.argmax(x, axis=1)
        m1 = K.one_hot(m1, 3)
        x = x * (1 - m1)
        
        fourth = K.max(x, axis=1)
        m1 = K.argmax(x, axis=1)
        m1 = K.one_hot(m1, 3)
        x = x * (1 - m1)
        
        fifth = K.max(x, axis=1)
        
        out = K.stack([first, second, third, fourth, fifth])
        out = K.stack([fifth])
        #print(out)
        #out.eval()
        response = K.mean(K.transpose(out), axis=1)
        
        return response

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], 3])
        #return input_shape

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return tuple([input_shape[0], 3])


# In[46]:
def MU(X,Y):
    print(K.max(X))

import theano
import theano.tensor as T
l2factor = 1e-5
l1factor = 2e-7
outdim=200
outdim2=500

def createModel():
    inputs = Input(shape=(faces_per_image, feature_per_face))
    dense_1 = Dense(1024,name='dense_1')(inputs)
    act1 = Activation('relu', name="relu1")(dense_1)
    dense_2 = Dense(512,name='dense_2')(act1)
    act2 = Activation('relu', name="relu2")(dense_2)
    
    dense_3 = Dense(128,name='dense_3')(act2)
    act3 = Activation('relu', name="relu3")(dense_3)
    
    dense_4 = Dense(3,name='dense_4')(act3)
    sigmoid = Activation("softmax",name="softmax1")(dense_4)
    
    prediction = Flatten(name='flatten')(sigmoid)
    MU(prediction,sigmoid)
    #Returns a 10 dim matrix
    prediction = SortLayer(k=1, label=1, name='output')(prediction)
    model = Model(inputs=inputs, outputs=prediction)
    return model

#def createModel2():
#    inputs = Input(shape=(faces_per_image, feature_per_face))
#   
#    dense_1 = Dense(outdim,name='dense_1',kernel_regularizer=l1_l2(l1=l1factor, l2=l2factor))(inputs)
#    act1 = Activation("relu",name="relu1")(dense_1)
#    dense_2 = Dense(outdim2,name='dense_2',kernel_regularizer=l1_l2(l1=l1factor, l2=l2factor))(act1)
#    act2 = Activation("relu",name="relu2")(dense_2)
#
#    dense_3 = Dense(1,name='dense_3',kernel_regularizer=l1_l2(l1=l1factor, l2=l2factor))(act2)
#    sigmoid = Activation("sigmoid",name="sigmoid1")(dense_3)
#    prediction = Flatten(name='flatten')(sigmoid)
#    prediction = SortLayer(k=1, label=1, name='output')(prediction)
#    model = Model(inputs=inputs, outputs=prediction)
#
#    return model
#
#def createModel3():
#    inputs = Input(shape=(faces_per_image, feature_per_face))
#   
#
#    dense_3 = Dense(1,name='dense_3',kernel_regularizer=l1_l2(l1=l1factor, l2=l2factor))(inputs)
#    sigmoid = Activation("sigmoid",name="sigmoid1")(dense_3)
#    prediction = Flatten(name='flatten')(sigmoid)
#    prediction = SortLayer(k=1, label=1, name='output')(prediction)
#    model = Model(inputs=inputs, outputs=prediction)
#    return model

a= x[0:200,:]
b= x[200:400,:]
c= x[400:600,:]
d= x[600:800,:]
e= x[800:1000,:]

a_l= y[0:200,:]
b_l= y[200:400,:]
c_l= y[400:600,:]
d_l= y[600:800,:]
e_l= y[800:1000,:]

score=np.zeros(5)

for i in range(5): 
    print(i)   
    m_name='model'+str(i+1)+'vip.h5'
    model=createModel()
    model.summary()
    if i==0:
        x_train=np.concatenate((b,c,d,e),axis=0)
        y_train=np.concatenate((b_l,c_l,d_l,e_l),axis=0)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 
        
        checkpoint = keras.callbacks.ModelCheckpoint(m_name, monitor='acc',
                                               save_best_only=True, save_weights_only=True, verbose=1,mode='max')
        early_stopping=keras.callbacks.EarlyStopping(monitor='acc', patience=200, verbose=0, mode='max')
        
        model.fit(x_train, y_train, epochs=1000,batch_size=1,verbose=1,callbacks=[early_stopping,checkpoint])
        scores = model.evaluate(a,a_l, verbose=0)
        score[i]=scores
        model.save(m_name)
    if i==1:
        x_train=np.concatenate((a,c,d,e),axis=0)
        y_train=np.concatenate((a_l,c_l,d_l,e_l),axis=0)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 
        
        checkpoint = keras.callbacks.ModelCheckpoint(m_name, monitor='acc',
                                               save_best_only=True, save_weights_only=True, verbose=1,mode='max')
        early_stopping=keras.callbacks.EarlyStopping(monitor='acc', patience=200, verbose=0, mode='max')
        
        model.fit(x_train, y_train, epochs=1000,batch_size=50,verbose=1,
                      callbacks=[checkpoint,early_stopping])
        scores = model.evaluate(b,b_l, verbose=0)
        score[i]=scores
        model.save(m_name)
    if i==2:
        x_train=np.concatenate((a,b,d,e),axis=0)
        y_train=np.concatenate((a_l,b_l,d_l,e_l),axis=0)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 
        
        checkpoint = keras.callbacks.ModelCheckpoint(m_name, monitor='acc',
                                               save_best_only=True, save_weights_only=True, verbose=1,mode='max')
        early_stopping=keras.callbacks.EarlyStopping(monitor='acc', patience=200, verbose=0, mode='max')
        
        model.fit(x_train, y_train, epochs=1000,batch_size=50,verbose=1,
                      callbacks=[checkpoint,early_stopping])
        scores = model.evaluate(c,c_l, verbose=0)
        score[i]=scores
        model.save(m_name)
    if i==3:
        x_train=np.concatenate((a,b,c,e),axis=0)
        y_train=np.concatenate((a_l,b_l,c_l,e_l),axis=0)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 
        
        checkpoint = keras.callbacks.ModelCheckpoint(m_name, monitor='acc',
                                               save_best_only=True, save_weights_only=True, verbose=1,mode='max')
        early_stopping=keras.callbacks.EarlyStopping(monitor='acc', patience=200, verbose=0, mode='max')
        
        model.fit(x_train, y_train, epochs=1000,batch_size=50,verbose=1,
                      callbacks=[checkpoint,early_stopping])
        scores = model.evaluate(d,d_l, verbose=0)
        score[i]=scores
        model.save(m_name)
    if i==4:
        x_train=np.concatenate((a,b,c,d),axis=0)
        y_train=np.concatenate((a_l,b_l,c_l,d_l),axis=0)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 
        
        checkpoint = keras.callbacks.ModelCheckpoint(m_name, monitor='acc',
                                               save_best_only=True, save_weights_only=True, verbose=1,mode='max')
        early_stopping=keras.callbacks.EarlyStopping(monitor='acc', patience=200, verbose=0, mode='max')
        
        model.fit(x_train, y_train, epochs=1000,batch_size=50,verbose=1,
                      callbacks=[checkpoint,early_stopping])
        scores = model.evaluate(e,e_l, verbose=0)
        score[i]=scores
        model.save(m_name)
    
