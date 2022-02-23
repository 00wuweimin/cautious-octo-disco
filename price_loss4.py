import torch
import torch.nn as nn
from torch.nn import RNNBase
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

##################################################
# #生成输入矩阵
# data = pd.read_csv('E:\\quant_research\\price_prediction\\price.csv')
# data_np = np.array(data)[:,:-1]
# #生成输入矩阵input_mat   batch*time*feature
# input_mat = []
# #时间维度长度
# k = data_np.shape[1]
# #时间特征
# feature2 = np.arange(k-1,-1,-1)
# for i in np.arange(data_np.shape[0]):
#     feature1 = data_np[i,:]
#     feature = np.vstack((feature1,feature2)).T
#     input_mat.append(feature)
# input_mat = np.array(input_mat)
# #存储
# np.save("E:\\quant_research\\price_prediction\\input.npy",input_mat)

#导入输入矩阵   batch*time*feature
input_mat = np.load("E:\\quant_research\\price_prediction\\input.npy")
#转化为tensor
input_tensor = torch.from_numpy(input_mat).type(torch.FloatTensor)

############################################
#定义RNN
class RNN(nn.Module):
    #num_classes表示最终每个样本输出的特征数
    def __init__(self):
        super(RNN, self).__init__()
        #定义RNN网络,输入单个数字.隐藏层size为[feature, hidden_size]
        self.rnn1 = nn.RNN(
            input_size = 2,  #input_size = 2
            hidden_size = 100,  #hidden_1_size=100
            num_layers = 1,
            batch_first = True  #输入形状为[batch_size, time_step, feature]
        )
        self.rnn2 = nn.RNN(
            input_size=100,  #hidden_1_size=100
            hidden_size=60,  #hidden_2_size=30
            num_layers=1,
            batch_first=True  # 输入形状为[batch_size, time_step, feature]
        )

        self.rnn3 = nn.RNN(
            input_size=60,  #hidden_1_size=100
            hidden_size=30,  #hidden_2_size=30
            num_layers=1,
            batch_first=True  # 输入形状为[batch_size, time_step, feature]
        )

        # 定义一个全连接层
        self.out1 = nn.Linear(30, 10)   #output_size=2
        self.out2 = nn.Linear(10, 2)  # output_size=2


    def forward(self, x):
        # 给定一个序列x,每个x.size=[batch_size, feature].同时给定一个h_state初始状态,RNN网络输出结果并同时给出隐藏层输出
        out_put1, final_timestep_1 = self.rnn1(x,None)
        out_put2, final_timestep_2 = self.rnn2(out_put1,None)
        out_put3, final_timestep_3 = self.rnn3(out_put2, None)
        #out_put2 的 shape 为[sample_size,time,hidden_2_size]
        out1 = self.out1(out_put3)
        out = self.out2(out1)
        return out


#调用RNN网络
RNN_USE = RNN()

#选择优化器,定义损失函数
LR = 0.01  #学习率
optimizer = torch.optim.Adam(RNN_USE.parameters(),lr=LR)
loss_func = nn.MSELoss(reduction='mean')

#######################################
#使用损失函数4
data = pd.read_csv('E:\\quant_research\\price_prediction\\price.csv')
data_np = np.array(data)[:,:-1]
k = data_np.shape[1]
#使用data_np做差分项
num1 = data_np[:,0:k-1]
num2 = data_np[:,1:k]
delta_num = num2 - num1
delta_num = torch.from_numpy(delta_num).type(torch.FloatTensor)
data_tensor = torch.from_numpy(np.array(data)).type(torch.FloatTensor)
p = data_tensor[:,-1]

#################################################
#评价预测效果
# #生成输入矩阵
# data_pre = pd.read_csv('E:\\quant_research\\price_prediction\\price_pre.csv')
# data_pre_np = np.array(data_pre)[:,:-1]
# #生成输入矩阵input_mat   batch*time*feature
# input_mat_pre = []
# #时间维度长度
# k_pre = data_pre_np.shape[1]
# #时间特征
# feature2_pre = np.arange(k_pre-1,-1,-1)
# for i in np.arange(data_pre_np.shape[0]):
#     feature1_pre = data_pre_np[i,:]
#     feature_pre = np.vstack((feature1_pre,feature2_pre)).T
#     input_mat_pre.append(feature_pre)
# input_mat_pre = np.array(input_mat_pre)
# #存储
# np.save("E:\\quant_research\\price_prediction\\input_pre.npy",input_mat_pre)

#导入输入矩阵   batch*time*feature
input_mat_pre = np.load("E:\\quant_research\\price_prediction\\input_pre.npy")
#转化为tensor
input_pre_tensor = torch.from_numpy(input_mat_pre).type(torch.FloatTensor)

#使用损失函数4
data_pre = pd.read_csv('E:\\quant_research\\price_prediction\\price_pre.csv')
data_pre_np = np.array(data_pre)[:,:-1]
k_pre = data_pre_np.shape[1]
#使用data_np做差分项
num1_pre = data_pre_np[:,0:k_pre-1]
num2_pre = data_pre_np[:,1:k_pre]
delta_num_pre = num2_pre - num1_pre
delta_num_pre = torch.from_numpy(delta_num_pre).type(torch.FloatTensor)
data_pre_tensor = torch.from_numpy(np.array(data_pre)).type(torch.FloatTensor)
p_pre = data_pre_tensor[:,-1]

############################################################
#训练神经网络
#损失函数列表
LOSS = []
#预测误差
LOSS_pre = []
#训练集预测效果
rate_train = []
#测试集预测效果
rate_test = []
#对训练集中的样本进行num次迭代
num = 2000
for step in np.arange(num):
    x = Variable(input_tensor)
    out = RNN_USE(x)
    # 使用神经网络输出值
    output1 = out[:, :, 0]
    output1 = output1[:, :-1]
    output2 = out[:, :, 1]
    loss = 0
    # 生成0 tensor,计算0与mul的MSE
    zero = np.zeros((p.shape))
    zero_tensor = torch.from_numpy(zero).type(torch.FloatTensor)
    for j in np.arange(k):
        output1_use = output1[:,j:]
        delta_num_use = delta_num[:,j:]
        output2_use = output2[:,j]
        mul = torch.sum((output1_use*delta_num_use),dim=1) + output2_use - p
        loss += loss_func(mul, zero_tensor)

    LOSS.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('训练集第%d次迭代'%(step))
    print('训练集损失值',loss)

    # 进行预测
    x_pre = Variable(input_pre_tensor)
    out_pre = RNN_USE(x_pre)
    # 使用神经网络输出值
    output1_pre = out_pre[:, :, 0]
    output1_pre = output1_pre[:, :-1]
    output2_pre = out_pre[:,:,1]
    loss_pre = 0
    # 生成0 tensor,计算0与mul的MSE
    zero_pre = np.zeros((p_pre.shape))
    zero_pre_tensor = torch.from_numpy(zero_pre).type(torch.FloatTensor)
    for j in np.arange(k_pre):
        output1_pre_use = output1_pre[:,j:]
        delta_num_pre_use = delta_num_pre[:,j:]
        output2_pre_use = output2_pre[:,j]
        mul_pre = torch.sum((output1_pre_use*delta_num_pre_use),dim=1) + output2_pre_use - p_pre
        loss_pre += loss_func(mul_pre, zero_pre_tensor)

    print('测试集损失', loss_pre)
    LOSS_pre.append(loss_pre)

    if step % 10 ==0:
    # 存储训练参数
        torch.save(RNN_USE.state_dict(), 'E:\\quant_research\\price_prediction\\loss4\\train2\\net\\net{}.pkl'.format(step))
# #提取训练参数,导入net2400参数
# RNN_USE.load_state_dict(torch.load('E:\\quant_research\\price_prediction\\loss4\\train2\\net\\net1990.pkl'))

np.save('E:\\quant_research\\price_prediction\\loss4\\train2\\loss_train.npy',LOSS)
np.save('E:\\quant_research\\price_prediction\\loss4\\train2\\loss_test.npy',LOSS_pre)


LOSS = np.load('E:\\quant_research\\price_prediction\\loss4\\train2\\loss_train.npy',allow_pickle=True)
LOSS_pre = np.load('E:\\quant_research\\price_prediction\\loss4\\train2\\loss_test.npy',allow_pickle=True)

plt.plot(LOSS[1000:],color="red")
plt.plot(LOSS_pre[1000:],color='blue')
plt.show()










