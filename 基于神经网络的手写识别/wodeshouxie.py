import numpy

import scipy.special

import matplotlib.pyplot

# ensure the plots are inside this notebook, not an external window 
%matplotlib inline

class neuralNetwork :
        
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate) :
        
        
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        
        self.lr = learningrate
        
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    
    def train(self,inputs_list,targets_list) :
        
        inputs = numpy.array(inputs_list,ndmin = 2).T
        targets = numpy.array(targets_list,ndmin = 2).T
        
 
        hidden_inputs = numpy.dot(self.wih, inputs) #计算隐藏层中的信号
        hidden_outputs = self.activation_function(hidden_inputs) #计算隐藏层中出现的信号

        final_inputs = numpy.dot(self.who, hidden_outputs) #将信号计算到最终输出层
        final_outputs = self.activation_function(final_inputs) #计算从最终输出层发出的信号

        output_errors = targets - final_outputs #输出层错误是（目标-实际）
        hidden_errors = numpy.dot(self.who.T, output_errors) #隐藏层错误是输出错误，按权重分割，在隐藏节点重新组合

        #更新隐藏层和输出层之间链接的权重                              
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        #更新输入层和隐藏层之间链接的权重
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    
    #查询神经网络
    def query(self, inputs_list):
        
        #将输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin = 2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs) #计算隐藏层中的信号
        hidden_outputs = self.activation_function(hidden_inputs) #计算隐藏层中出现的信号

        final_inputs = numpy.dot(self.who, hidden_outputs) #将信号计算到最终输出层
        final_outputs = self.activation_function(final_inputs) #计算从最终输出层发出的信号

        return final_outputs 



#输入、隐藏、输出节点数
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

#学习率
learning_rate = 0.1

#创建神经网络实例
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#将文件导入列表
training_data_file = open("E:\pzy的学习文档\python 神经网络\mnist_test.csv", 'r') 
training_data_list = training_data_file.readlines()
training_data_file.close() 

#epochs是训练数据集用于训练的次数
epochs = 5
for e in range(epochs):
    
    #查看培训数据集中所有数据
    for record in training_data_list:  
        all_values = record.split(',') #用逗号拆分记录
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  #将输入颜色值范围进行缩放
        targets = numpy.zeros(output_nodes) + 0.01 #创建目标输出值
        #所有值【0】都是此纪录的目标标签
        targets[int(all_values[0])] = 0.99 
        n.train(inputs, targets) 
        pass
    pass

#将文件导入列表
test_data_file = open("E:\pzy的学习文档\8.png", 'rb') 
test_data_list = test_data_file.readlines() 
test_data_file.close()

#测试自己家手写的



+
 all_values = record.split(',')
 print(all_values[0])      
#检查测试数据集中所有记录
scorecard = []
for record in test_data_list:

    correct_label = int(all_values[0]) #正确答案是第一个值
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #缩放和移动输入
    outputs = n.query(inputs) #查询网络
    label = numpy.argmax(outputs) #最大值的索引对应标签
    #追加正确或不正确的列表
    if (label == correct_label):
        scorecard.append(1) #正确记分卡加1
    else:
        scorecard.append(0) #错误记分卡加0
        pass
    pass

#计算成绩分数，正确答案的分数
scorecard_array = numpy.asarray(scorecard) 
print ("performance = ", scorecard_array.sum() / scorecard_array.size)