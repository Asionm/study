import torch
import torch.nn as nn
from torch.nn import Sequential
import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision.models as models
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


class CNN_Vir:
    def __init__(self, path, classes, batch_size=32, epochs=20,
                 lr=0.001, device="gpu", opt=0, lossf=0, model_type=0,
                 load=False):
        self.model_type = model_type
        #数据集路径
        self.path = path
        # 车辆标签
        self.classes = classes
        # 批量
        self.batch_size = batch_size
        # 迭代次数
        self.epochs = epochs
        # 学习率
        self.lr = lr
        # 加载器与数据集
        self.train_loader, self.test_loader ,\
            self.train_dataset, self.test_dataset = self.pre_data()
        self.model = None
        self.device = device
        self.opt = opt
        self.lossf = lossf
        self.epoch_loss = []
        self.load = load

    #预处理数据 返回两个loader 和数据集
    def pre_data(self):
        # 一系列的图片预处理可以由此定义一个对象来处理
        if self.model_type == 0:
            size = 32
        else:
            size = 224
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            # 将图片转为tensor
            transforms.ToTensor(),
            # 调整图片大小
            # 在图片的中间区域进行裁剪
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            # 图片的标准化处理均值和标准差均设为0.5
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.2, 0.2, 0.2]
            )
        ])

        # 读取所有图像，利用transform处理
        full_data = torchvision.datasets.ImageFolder(root=self.path,
                                                     transform=transform)
        # 设置训练集和测试集比例
        # 这里train_size是一个长度矢量，并非是比例，我们将训练和测试进行8/2划分
        train_size = int(len(full_data) * 0.8)
        test_size = len(full_data) - train_size

        # 数据集划分
        train_dataset, test_dataset = random_split(full_data, [train_size, test_size])

        # 设置数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        #返回加载器
        return train_loader, test_loader, train_dataset, test_dataset

    # 返回可自动梯度求导的变量
    # 这里这样设置是为了使用gpu
    def get_variable(self, x):
        x = torch.autograd.Variable(x)
        if self.device == "gpu":
            return x.cuda()
        elif self.device == "cpu":
            return x
        else:
          return x.cuda() if torch.cuda.is_available() else x

    #自建模型
    def cnn_tr(self):
        # 设置卷积神经网络
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                # 二维卷积层
                self.conv1 = nn.Conv2d(3, 20, 5, 1)
                # 二维卷积层
                self.conv2 = nn.Conv2d(20, 40, 4, 1)
                # 二维卷积层
                self.conv3 = nn.Conv2d(40, 60, 3, 1)
                # 全连接层
                self.dense = Sequential(
                    nn.Linear(1 * 1 * 60, 400),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(400, 10),
                )
            # N = (W - F + 2P)/S + 1
            def forward(self, x):
                x = F.relu(self.conv1(x))  # 220x220x20
                x = F.max_pool2d(x, 2, 2)  # 110x110x20
                x = F.relu(self.conv2(x))  # 107x107x40
                x = F.max_pool2d(x, 2, 2)  # 53x53x40
                x = F.relu(self.conv3(x))  # 51x51x60
                x = F.max_pool2d(x, 2, 2)  # 25x25x60
                #降维处理
                x = x.view(-1, 1 * 1 * 60)
                x = self.dense(x)
                return x
        # 训练
        cnn = CNN()
        if self.device == "gpu":
            cnn = cnn.cuda() # 利用gpu训练
        else:
            cnn = cnn

        if self.load:
            return cnn

        # 损失函数
        if self.lossf == 0:
            lossF = nn.CrossEntropyLoss()
        elif self.lossf == 1:
            lossF = nn.NLLoss2d()
        elif self.lossf == 2:
            lossF = nn.NLLLoss2d()


        # 优化器
        if self.opt == 0:
            optimizer = torch.optim.Adam(cnn.parameters(), lr=self.lr)
        elif self.opt == 1:
            optimizer = torch.optim.Adagrad(cnn.parameters(), lr=self.lr)
        elif self.opt == 2:
            optimizer = torch.optim.NAdam(cnn.parameters(), lr=self.lr)
        elif self.opt == 3:
            optimizer = torch.optim.SGD(cnn.parameters(), lr=self.lr)

        # 设置这个可以防止利用gpu时报错
        torch.backends.cudnn.enable = True
        torch.backends.cudnn.benchmark = True

        #定义初始损失
        loss_pth = -float("inf")
        i_pth = 0
        epoch_loss = []
        # 训练
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_correct = 0.0
            print("Epochs [{}/{}]".format(epoch, self.epochs))
            # 批量训练
            for data in tqdm(self.train_loader):
                X_train, y_train = data
                #转为gpu格式
                X_train, y_train = self.get_variable(X_train), self.get_variable(y_train)
                outputs = cnn(X_train)
                #_表示真正的值，predict表示标签值
                _, predict = torch.max(outputs.data, 1)
                optimizer.zero_grad()
                loss = lossF(outputs, y_train)
                loss.backward()
                optimizer.step()
                #记录损失值与成功个数
                running_loss += loss.item()
                running_correct += torch.sum(predict == y_train.data)
            epoch_loss.append(running_loss)


        self.epoch_loss = epoch_loss
        self.model = cnn

    def model_resent50(self):
        my_resnet50 = models.resnet50(pretrained=True)

        # 由于预训练的模型中的大多数参数已经训练好了，因此先将模型参数的自动梯度求解设为false
        for param in my_resnet50.parameters():
            param.requires_grad = False

        # 将resnet50最后一层输出的类别数，改为ant-bee数据集的类别数，修改后改成梯度计算会恢复为默认的True
        fc_inputs = my_resnet50.fc.in_features
        my_resnet50.fc = nn.Sequential(nn.Linear(fc_inputs, 10))
        # 以上操作相当于固定网络全连接层之前的参数，只训练全连接层的参数

        # 损失函数
        if self.lossf == 0:
            lossF = nn.CrossEntropyLoss()
        elif self.lossf == 1:
            lossF = nn.NLLLoss()
        elif self.lossf == 2:
            lossF = nn.NLLLoss2d()


        # 优化器
        if self.opt == 0:
            optimizer = torch.optim.Adam(my_resnet50.parameters())
        elif self.opt == 1:
            optimizer = torch.optim.Adagrad(my_resnet50.parameters())
        elif self.opt == 2:
            optimizer = torch.optim.NAdam(my_resnet50.parameters())
        elif self.opt == 3:
            optimizer = torch.optim.SGD(my_resnet50.parameters(), lr=self.lr)

        if self.device == "gpu":
            my_resnet50 = my_resnet50.cuda()
        if self.load:
            return my_resnet50


        my_resnet50.train()
        epoch_loss = []
        for epoch in range(self.epochs):
            losses = []
            for inputs, outputs in tqdm(self.train_loader):
                optimizer.zero_grad()
                # 前向传播
                inputs = self.get_variable(inputs)
                outputs = self.get_variable(outputs)
                results = my_resnet50(inputs)
                # loss计算
                loss = lossF(results, outputs)
                losses.append(loss.item())
                # 梯度计算
                loss.backward()
                # 参数更新
                optimizer.step()
            epoch_loss.append(torch.tensor(losses).mean().item())
            print("epoch: {0} Loss: {1}".format(epoch, epoch_loss[epoch]))
            self.model = my_resnet50
            self.epoch_loss = epoch_loss

    #绘图
    def draw(self, type="iter"):
        if self.epoch_loss:
            plt.title("loss vs. Number of Training Epochs")
            plt.xlabel("Training Epochs")
            plt.ylabel("Loss")
            plt.plot(range(self.epochs), self.epoch_loss, label="train")
            plt.xticks(np.arange(0, self.epochs))
            plt.legend()
            plt.show()
        else:
            raise "Please train first!"



    #模型评价
    def evaluate(self):

        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score, recall_score, f1_score

        from tqdm import tqdm
        label_t = torch.Tensor()
        predicted_t = torch.Tensor()
        self.model.eval()
        for i, data in enumerate(tqdm(self.test_loader), 0):
            inputs, labels = data
            inputs = self.get_variable(inputs)
            labels = self.get_variable(labels)
            outputs = self.model(inputs)
            image, predicted = torch.max(outputs, 1)
            labels = labels.cpu()
            predicted = predicted.cpu()
            label_t = torch.cat((label_t, labels), 0)
            predicted_t = torch.cat((predicted_t, predicted), 0)

        acc = accuracy_score(label_t, predicted_t)
        p = precision_score(label_t, predicted_t, average='macro')
        r = recall_score(label_t, predicted_t, average='macro')
        f1score = f1_score(label_t, predicted_t, average='macro')
        print('The testing set Accuracy of the network is:', acc)
        print('The testing set Precision of the network is:', p)
        print('The testing set Recall of the network is:', r)
        print('The testing set F-score of the network is:', f1score)

    #保存训练模型
    def save(self, name="model.pth"):
        torch.save(self.model.state_dict(), name)

    #加载模型
    def load_model(self, name="model.pth"):
        checkpoint = torch.load(name)
        if self.model_type == 0:
            self.model = self.cnn_tr()
            self.model.load_state_dict(checkpoint)
        elif self.model_type == 1:
            self.model = self.model_resent50()
            self.model.load_state_dict(checkpoint)

        if self.device == "gpu":
            self.model = self.model.cuda()

    #主程序
    def run(self):
        if self.model_type == 0:
            self.cnn_tr()
        elif self.model_type == 1:
            self.model_resent50()


#定义数据集路径
path = "./train/"
#车辆种类
classes = ["bus", "family sedan", "fire engine", "heavy truck", "jeep",
           "minibus", "racing car", "SUV", "taxi", "truck"]

test = CNN_Vir(path, classes, device="gpu", model_type=1, epochs=3, batch_size=32, lossf=1)
test.run()
test.evaluate()
