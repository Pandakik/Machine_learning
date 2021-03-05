# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:49:59 2020

@author: daxiguammm
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:42:42 2020

@author: daxiguammm
"""
#1
#Dropout
#建立模型
model = tf.keras.Sequential()
model.add(ft.keras.layers.Flatten(input_shape=(28,28)))#输入变成28*28的向量
model.add(tf.keras.layers.Dense(128,activation='relu'))#隐藏层
model.aad(tf.keras.layers.Dropout(0.5))#丢弃掉50%
model.add(tf.keras.layers.Dense(128,activation='relu'))#隐藏层
model.aad(tf.keras.layers.Dropout(0.5))#丢弃掉50%
model.add(tf.keras.layers.Dense(128,activation='relu'))#隐藏层
model.aad(tf.keras.layers.Dropout(0.5))#丢弃掉50%
model.add(tf.keras.layers.Dense(10,activation='softmax'))#长度为10的概率输出

#2
#减小网络规模抑制过拟合
model = tf.keras.Sequential()
model.add(ft.keras.layers.Flatten(input_shape=(28,28)))#输入变成28*28的向量
model.add(tf.keras.layers.Dense(32,activation='relu'))#隐藏层
model.add(tf.keras.layers.Dense(10,activation='softmax'))#长度为10的概率输出


#3正则化 控制参数规模







