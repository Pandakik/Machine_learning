# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:26:30 2020

@author: daxiguammm
"""

#softmax分类
'''
对数几率回归解决的是二分类的问题
对于多个选项的问题，我们可以使用softmax函数
它是对数几率回归在N个可能不同的值上的推广

softmax个样本分量之和为1
当只有两个类别时，与对数几率回归完全相同
'''

#tf.keras交叉熵
'''
在tf.keras里，对于多分类问题我们使用
    categorical_crossentropy和
    sparse_categorical_crossentropy
来计算softmax交叉熵    
'''

#Fashion MNIST数据集
'''
Fasion MNIST的作用时成为经典MNIST数据集的简易替换，
MNIsT数据集包含手写数字(0,1,2等)的图像，这些图像-
-的格式与本节课中使用的服饰图像的格式相同

包含七万张图片
使用六万张图片训练网络，并使用一万张图片评估经过学习的网络分类图像的准确率

可以从TensorFlow直接访问Fasion MNIST,只需导入和加载数据即可
'''