# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:42:42 2020

@author: daxiguammm
"""
#1
#顺序编码
import tensorflow as tf
import pamdas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline#使图像能够直接显示在页面上

#加载数据
(traim_image,train_lable), (test_image,test_lable)= tf.keras.datasets.fashion_mnist.load_data()
#其中 train_lable 为该图片的分类标号

train_image.shape#查看大小
test_image.shape#查看大小
plt.imshow(train_image[0])#画第一张图片
np.max(train_image[0])#查看最大值

#归一化
traim_image=traim_image/255
test_image=test_image/255

#建立模型
model = tf.keras.Sequential()
model.add(ft.keras.layers.Flatten(input_shape=(28,28)))#输入变成28*28的向量
model.add(tf.keras.layers.Dense(128,activation='relu'))#隐藏层
model.add(tf.keras.layers.Dense(10,activation='softmax'))#长度为10的概率输出

#训练
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',#顺序标签使用这个
              metrics=['acc']
)

model.fit(train_image,train_lable,epochs=5)
model.evaluate(test_image,test_lable)

#2
#独热编码
train_lable#为顺序编码：0 1 5 6 7 9....
'''
使用 categorical_crossentropy
北京：[1,0,0]
上海：[0,1,0]
深圳：[0,0,1]
'''
#转化为独热编码
train_lable_onehot = tf.keras.utils.to_categorical(train_lable)
train_label_onehot[0]#查看数据
test_lable_onehot = tf.keras.utils.to_categorical(test_lable)

#建立模型
model = tf.keras.Sequential()
model.add(ft.keras.layers.Flatten(input_shape=(28,28)))#输入变成28*28的向量
model.add(tf.keras.layers.Dense(128,activation='relu'))#隐藏层
model.add(tf.keras.layers.Dense(10,activation='softmax'))#长度为10的概率输出

#训练
model.compile(optimizer='adam',
              loss='categorical_crossentropy',#顺序标签使用这个
              metrics=['acc']
)



model.fit(train_image,train_lable_onehot,epochs=5)

#预测
predict=model.predict(test_image)
predict.shape

predict[0]
np.argmax(predict[0])
test_lable[0]














