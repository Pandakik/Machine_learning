# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:16:45 2020

@author: daxiguammm
"""

#过拟合
'''
在训练数据上得分很高，在测试数据上得分相对比较低
'''
#欠拟合
'''
在训练数据上得分比较低，在测试数据上得分相对比较低
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

#训练
model.compile(optimizer='adam',
              loss='categorical_crossentropy',#顺序标签使用这个
              metrics=['acc']
)



history=model.fit(train_image,train_lable_onehot,
          epochs=10，
          validation_data=(test_image, test_lable_onehot))# validation_data得到在test数据集上的正确率
#out：训练集上的loss和正确率，test数据集上的loss和正确率
#结果
'''
#经过7次训练后 val_loss出现了问题，不下降反而上升了，出现了过拟合
#train数据集和test数据集上正确率相差很大
'''

#Dorpout抑制过拟合
'''
人为丢弃一些层

（1）取平均的作用：先回到标准的模型即没有dropout，我们用相同的训练数据取训练5个不同的神经网络，一般会得到5个不同的结果
此时我们可以采用‘五个结果取均值’或者‘多数取胜的投票策略’取决定最终结果
（2）减少神经元之间复杂的共适应关系：因为fropout程序导致两个神经元
不一定每次都在一个dropout网络中出现。这样权值的更新不再依赖于有固定关系的隐含节点
的共同作用，阻止了某些特征仅仅在其他特定特征下才有效果的情况。
（3）Dopout类似于性别在生物进化中的角色：
物种为了生存往往会倾向于适应这种环境，环境突变则会导致物种难以做出及时反应，性别的出现可以繁衍出适应
新环境的变种，有效的组织过拟合，即避免环境改变时物种可能面临的灭绝
'''

#参数选择原则
'''
理想的模型时刚好在欠拟合和过拟合的界线上，也就是刚好拟合数据。

首先开发一个过拟合模型：
（1）添加更多的层
（2）让每一层变得更大
（3）训练更多轮

抑制过拟合
（1）dropout
（2）正则化
（3）图像增强

再次，调节超参数：学习速率，隐藏层单元数，训练轮次

最好抑制过拟合办法：增加训练数据

'''

#构建网络的总原则
'''
总的原则是：保证神经网络容量足够拟合数据
一、增大网络容量，直到过拟合
二、采取措施抑制过拟合
三、继续增大网络容量，直到过拟合
'''












