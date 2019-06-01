# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:43:11 2018

@author: Vijay Gupta
"""

import os

def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)
        
def link(src,des):
    if not os.path.exists(des):
        os.symlink(src,des,target_is_directory=True)
        
mkdir('E:/fruits 360 cnn/fruits-360/fruits_360_small1')

classes=['Kiwi','Banana','Raspberry']

train_path_from=os.path.abspath('E:/fruits 360 cnn/fruits-360/Training')
train_path_to=os.path.abspath('E:/fruits 360 cnn/fruits-360/fruits_360_small1/Training')
test_path_to=os.path.abspath('E:/fruits 360 cnn/fruits-360/fruits_360_small1/Test')

mkdir(train_path_to)
mkdir(test_path_to)

for c in classes:
    link(train_path_from+'/'+c,train_path_to+'/'+c)