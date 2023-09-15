# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 00:41:10 2022

@author: Goutham
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 22:07:19 2021

@author: Dragon_Slayer
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(32,3,3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(128,activation='relu'))

classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'data/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'data/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
classifier.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


