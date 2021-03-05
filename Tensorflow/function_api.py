# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:06:53 2020

@author: daxiguammm
"""
#函数式API与多输入多输出模型
'''
函数式API类似于函数的调用，
把每一层看成一个函数来调用这一层
'''

#demo1
import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
fashion_mnist = keras.datasets.fashion_mnist
#labels
(traim_image,train_labels), (test_image,test_labels)=fashion_mnist.load_data()

#归一化
traim_image=traim_image/255
test_image=test_image/255

#函数式api
#设置输入
input = keras.Input(shape = (28,28))
x = keras.layers.Flatten()(input)#函数调用
x = keras.layers.Dense(32,activation='relu')(x)#32个输出单元
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(32,activation='relu')(x)#32个输出
output = keras.layers.Dense(10,activation='softmax')(x)

#建立模型
model = keras.Model(inputs=input,outputs=output)

#训练
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',#顺序标签使用这个
              metrics=['acc']
)

model.fit(train_image,train_lable,epochs=5)
model.evaluate(test_image,test_lable)


#demo2
#多输入单输出，判断这两个是不是同一个东西
input1 = keras.Input(shape = (28,28))
input2 = keras.Input(shape = (28,28))
x1 = keras.layers.Flatten()(input1)#函数调用
x2 = keras.layers.Flatten()(input2)#函数调用
x = keras.layers.concatenate([x1, x2])#合并x1,x2
x = keras.layers.Dense(32,activation='relu')(x)#32个输出单元
output = keras.layers.Dense(1,activation='sigmoid')(x)

model = keras.Model(inputs=[input1, input2],outputs=output)
#查看model
model.summary()


