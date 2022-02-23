# cautious-octo-disco
 derivative pricing model with RNN

Environment 
 Python 3.10
 Pytorch 1.10.2
 
Usage
 Test the model on data 'price_pre.csv'
 
Network
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
