import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
from sklearn.preprocessing import scale as sc
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

class Concrete_predict_withANN:


    def __init__(self, path='concrete_data_prepared.csv', label='strength',
                scale=0, batch_size=128, hidden_dim=8, lr=0.01,
                iteration_num=100, opt=0):
        self.path = path
        self.label = label
        self.batch_size = batch_size
        self.scale = scale
        self.hidden_dim = hidden_dim
        self.lr = lr #学习率
        self.iteration_num = iteration_num #迭代次数
        self.opt = opt
        #优化器选择 默认为adam
        #若为1则选择sgd
        #若为2则选择nadam
        #对象定义的同时进行训练
        self.model, self.loss_list, self.dataset = self.con_tr()
        #返回训练好的模型，以及损失记录，以及测试集(用于测试)

    #数据预处理
    def pre_data(self):
        # 注意此处类型转化为float，不然后面求导会报错
        train = pd.read_csv(self.path, dtype=np.float32)
        # 获取x，y
        y = train.loc[:, train.columns == self.label].values
        x = train.loc[:, train.columns != self.label].values

        #如果scale为0，则不数据标准化
        if self.scale == 0:
            x_data = x
        #如果scale为1，则用minmax标准化
        elif self.scale == 1:
            transfer = MinMaxScaler(feature_range=(0, 1))
            # 处理后的范围
            x_data = transfer.fit_transform(x)
            # 对data进行处理
        elif self.scale == 2:
            x_data = sc(X=x, with_mean=True, with_std=True, copy=True)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=2019)
        #转化为torch形式
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(y_test)

        train = torch.utils.data.TensorDataset(x_train, y_train)
        test = torch.utils.data.TensorDataset(x_test, y_test)
        '''
        DataLoader用于随机播放和批量处理数据。
        它可用于与多处理工作程序并行加载数据
        在dataset基础上多了batch_size, shuffle等操作
        '''
        train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=True)
        # 返回loader
        return train_loader, x.shape[1], {"train":[x_train,y_train], "test":[x_test,y_test]}
    #此函数为一个整体的函数用于训练

    def con_tr(self):
        train_loader, features, dataset = self.pre_data()
        # ANN全连接层神经网络
        class ANNModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(ANNModel, self).__init__()
                # 定义层
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.relu1 = nn.ReLU()  # nn.Linear为线性关系，加上激活函数转为非线性

                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.relu2 = nn.ReLU()

                self.fc3 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu1(out)
                out = self.fc2(out)
                out = self.relu2(out)
                out = self.fc3(out)
                return out
        input_dim = features  # 输入层神经元个数
        hidden_dim = self.hidden_dim  # hidden layer 神经元个数
        output_dim = 1  # 输出层神经元个数
        # 定义神经网络
        model = ANNModel(input_dim, hidden_dim, output_dim)
        # 用均方差作为损失函数
        MSELoss = nn.MSELoss()
        # 用于记录每次epcoh的损失值
        loss_list = []
        # 学习率
        learning_rate = self.lr
        # 优化器选择
        if self.opt == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif self.opt == 1:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif self.opt == 2:
            optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

        iteration_num = self.iteration_num  # 迭代次数
        # tqdm用于显示进度条
        for epoch in tqdm(range(iteration_num)):
            for fea, strength in train_loader:
                # 将其转化为变量
                train = Variable(fea)
                # 标签为强度值
                labels = Variable(strength)
                optimizer.zero_grad()
                outputs = model(train)
                loss = MSELoss(outputs, labels)
                loss.backward()
                optimizer.step()
            # 记录每次迭代的损失值
            loss_list.append(loss.item())
        #返回回模型与其训练过程的损失
        return model, loss_list, dataset

    #测试函数
    def predict(self, data=0, evaluate_result=True):
        if data == 0:
            data = self.dataset['test']
        elif data == 1:
            data = self.dataset['train']

        X_test = Variable(data[0])
        y_test = Variable(data[1])

        y_pred = self.model(X_test)
        MSELoss = nn.MSELoss()
        R2 = r2_score(y_pred.detach().numpy(), y_test.numpy())
        loss = MSELoss(y_pred, y_test)
        if evaluate_result:
            print("MSE均方差: ", loss)
            print("R2值: ", R2)
        return y_pred, R2, loss

    def draw(self, type):
        plt.rcParams['font.sans-serif'] = ['FangSong']
        #迭代图
        if type == 'iter':
            plt.xlabel('迭代次数', fontsize=15)
            plt.ylabel('损失值', fontsize=15)
            plt.title("迭代次数与损失值关系图", fontsize=15, loc='center', color='black')
            plt.plot([i for i in range(self.iteration_num)],
                     self.loss_list, label="LOSS")

        elif type == 'accuracy':
            y_test_pred, R2_test, loss_test = self.predict(evaluate_result=False)
            y_train_pred, R2_train, loss_train = self.predict(data=1, evaluate_result=False)
            #添加绘制overall的
            y_pred = torch.cat((y_train_pred ,y_test_pred))
            y = torch.cat((self.dataset['train'][1],self.dataset['test'][1]))
            R2_overall = r2_score(y_pred.detach().numpy(), y.numpy())

            plt.axis([0, 80, 0, 80])  # 设置轴的范围
            plt.scatter(y_test_pred.detach().numpy(), self.dataset['test'][1].numpy(),
                        s=25, c='green', marker='*', label='测试')
            plt.scatter(y_train_pred.detach().numpy(), self.dataset['train'][1].numpy(),
                        s=25, c='red', marker='+', label='训练')
            plt.legend(loc="upper left", fontsize=10)
            #画直线
            plt.plot([0,80],[0,80], c='blueviolet', linestyle='dashed')
            plt.xlabel('预测值', fontsize=15)
            plt.ylabel('真实值', fontsize=15)
            plt.title("预测值与真实值比较图", fontsize=15, loc='center', color='black')
            plt.text(70, 5, 'R2_overall={:.2f}\nR2_test={:.2f}\nR2_train={:.2f}\nMSE_test={:.2f}\nMSE_train={:.2f}'.
                     format(R2_overall,R2_test,R2_train,loss_test,loss_train), ha='center', fontsize=10, c='black')

        plt.show()
        plt.close()





test = Concrete_predict_withANN(iteration_num=1000, lr=0.01, scale=2, hidden_dim=300)
test.draw('iter')
test.draw('accuracy')