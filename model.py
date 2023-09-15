# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 02:39:33 2022

@author: Goutham
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout,BatchNormalization
from keras.models import load_model


#model
classifier = Sequential()

classifier.add(Conv2D(32,(3,3),input_shape=(24,24,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (1,1)))

classifier.add(Conv2D(32,(3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (1,1)))

classifier.add(Conv2D(64,3,3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (1,1)))

classifier.add(Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(128,activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(2,activation='softmax'))

classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

def Generator(dir, gen=ImageDataGenerator(rescale=1./255),shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical'):
    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)
    
batchSize = 32
targetSize = (24,24)
train_batch= Generator('data/train',shuffle=True, batch_size=batchSize,target_size=targetSize)
valid_batch= Generator('data/test',shuffle=True, batch_size=batchSize,target_size=targetSize)
stepsPerEpoch= len(train_batch.classes)//batchSize
validationSteps = len(valid_batch.classes)//batchSize
print(stepsPerEpoch,validationSteps)

classifier.fit(train_batch,steps_per_epoch=stepsPerEpoch, epochs=15,validation_data=valid_batch,validation_steps=validationSteps)

classifier.save('models/cnnnet.h5', overwrite=True)







