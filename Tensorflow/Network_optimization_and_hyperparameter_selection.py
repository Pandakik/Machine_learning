# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:01:00 2020

@author: daxiguammm
"""

#网络容量
'''
可以认为与网络中的可训练参数成正比
网络中的神经元数越多，层数越多，神经元的拟合能力越强
但是训练速度慢、难度越大，越容易产生过拟合。
'''

#如何选择超参数？
'''
所谓超参数，也就是搭建神经网络中，需要我们自己如选择（不是通过梯度下降算法去优化）的那些参数。
比如，中间层的神经元个数，学习速率。
'''

#如何提高网络的拟合能力
'''
一种显然的想法是增大网络容量：
1、增加层
2、增加隐藏神经元个数

单纯的增加神经元的个数对于网络性能的提高并不明显，
增加层会大大提高网络的拟合能力，这也是为什么现在
深度学习的层越来越深的原因。

注意：单层的神经元个数，不能太小，太小的话，会造成信息瓶颈，使得模型欠拟合
'''

#Demo
#建立模型
model = tf.keras.Sequential()
model.add(ft.keras.layers.Flatten(input_shape=(28,28)))#输入变成28*28的向量
model.add(tf.keras.layers.Dense(128,activation='relu'))#隐藏层

#增加隐藏层
model.add(tf.keras.layers.Dense(128,activation='relu'))#隐藏层
model.add(tf.keras.layers.Dense(128,activation='relu'))#隐藏层

model.add(tf.keras.layers.Dense(10,activation='softmax'))#长度为10的概率输出
#正确率到达91.03%



