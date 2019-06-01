# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:15:31 2018

@author: Vijay Gupta
"""

from keras.models import Model
from keras.layers import Dense,Flatten,Input,Lambda
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from glob import glob

IMAGE_SIZE=[100,100]

epochs=5
batch_size=32

train_path='E:/fruits 360 cnn/fruits-360/fruits_360_small1/Training'
test_path='E:/fruits 360 cnn/fruits-360/fruits_360_small1/Test'
image_files=glob(train_path+'/*/*.jpg')
test_files=glob(test_path+'/*/*.jpg')
folders=glob(train_path+'/*')

plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()

vgg=VGG16(input_shape=IMAGE_SIZE+ [3],weights='imagenet',include_top=False)

for layer in vgg.layers:
    layer.trainable=False
    
x=Flatten()(vgg.output)
prediction=Dense(len(folders),activation='softmax')(x)

model=Model(inputs=vgg.input,outputs=prediction)

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

gen=ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,
                       shear_range=0.1,zoom_range=0.2,horizontal_flip=True,vertical_flip=True,
                       preprocessing_function=preprocess_input)

test_gen=gen.flow_from_directory(test_path,target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels=[None]*len(test_gen.class_indices)
for k,v in test_gen.class_indices.items():
    labels[v]=k

train_generator=gen.flow_from_directory(train_path,target_size=IMAGE_SIZE,
                                        shuffle=True,batch_size=batch_size)


test_generator=gen.flow_from_directory(test_path,target_size=IMAGE_SIZE,
                                        shuffle=True,batch_size=batch_size)

r=model.fit_generator(train_generator,validation_data=test_generator,epochs=epochs,
                      steps_per_epoch=len(image_files),validation_steps=len(test_files))


    


