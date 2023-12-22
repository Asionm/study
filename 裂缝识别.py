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
import cv2
from PIL import Image
import matplotlib.ticker as ticker
import os
from torchvision.transforms import Compose, Normalize, ToTensor
os.environ['KMP_DUPLICATE_LIB_OK']='True'



# 工具类
class Tool:
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

    def im_convert(self, tensor):
        image = tensor.cpu().clone().detach().numpy()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
        image = image.clip(0, 1)
        return image

    def choose_cuda(self, cnn):
        if self.device == "gpu":
            return cnn.cuda()  # 利用gpu训练
        else:
            return cnn

    def choose_lossf(self):
        # 损失函数
        if self.lossf == 0:
            return nn.CrossEntropyLoss()
        elif self.lossf == 1:
            return nn.NLLoss2d()
        elif self.lossf == 2:
            return nn.BCEWithLogitsLoss()

    def choose_opt(self, cnn):
        # 优化器
        if self.opt == 0:
            return torch.optim.Adam(cnn.parameters(), lr=self.lr)
        elif self.opt == 1:
            return torch.optim.Adagrad(cnn.parameters(), lr=self.lr)
        elif self.opt == 2:
            return torch.optim.NAdam(cnn.parameters(), lr=self.lr)
        elif self.opt == 3:
            return torch.optim.SGD(cnn.parameters(), lr=self.lr)

    def train(self, cnn, optimizer, lossF):
        # 定义初始损失
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
                # 转为gpu格式
                X_train, y_train = self.get_variable(X_train), self.get_variable(y_train)
                outputs = cnn(X_train)
                # _表示真正的值，predict表示标签值
                _, predict = torch.max(outputs.data, 1)
                optimizer.zero_grad()
                loss = lossF(outputs, y_train)
                loss.backward()
                optimizer.step()
                # 记录损失值与成功个数
                running_loss += loss.item()
                running_correct += torch.sum(predict == y_train.data)
            epoch_loss.append(running_loss)

        self.epoch_loss = epoch_loss
        self.model = cnn

    # =====获取CAM start=====
    def makeCAM(self, feature, weights, classes_id):
        print(feature.shape, weights.shape, classes_id)
        # batchsize, C, h, w
        bz, nc, h, w = feature.shape
        cam = weights[classes_id].dot(feature.reshape(nc, h * w))
        cam = cam.reshape(h, w)  # (7, 7)
        # 归一化到[0, 1]之间
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        # 转换为0～255的灰度图
        cam_gray = np.uint8(255 * cam)
        # 最后，上采样操作，与网络输入的尺寸一致，并返回
        return cv2.resize(cam_gray, (224, 224))

    def getCam(self, model, pic_path):
        model.eval()

        image_transform = transforms.Compose([
            # 将输入图片resize成统一尺寸
            transforms.Resize([224, 224]),
            # 将PIL Image或numpy.ndarray转换为tensor，并除255归一化到[0,1]之间
            transforms.ToTensor(),
            # 标准化处理-->转换为标准正太分布，使模型更容易收敛
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        feature_data = []
        def feature_hook(model, input, output):
            if self.device == "gpu":
                feature_data.append(output.data.cpu().numpy())
            else:
                feature_data.append(output.data.numpy())
        if self.model_type == 1:
            model._modules.get('layer4').register_forward_hook(feature_hook)

            if self.device == "gpu":
                fc_weights = model._modules.get('fc')[0].weight.data.cpu().numpy()
            else:
                fc_weights = model._modules.get('fc')[0].weight.data.numpy()

        elif self.model_type == 2:
            model._modules.get('avgpool').register_forward_hook(feature_hook)
            if self.device == "gpu":
                fc_weights = model._modules.get('classifier')[6].weight.data.cpu().numpy()
            else:
                fc_weights = model._modules.get('classifier')[6].weight.data.numpy()

        # 获取预测类别id
        image = image_transform(Image.open(pic_path)).unsqueeze(0)
        if self.device == "gpu":
            out = model(image.cuda())
        else:
            out = model(image)

        if self.device == "gpu":
            predict_classes_id = np.argmax(F.softmax(out, dim=1).data.cpu().numpy())
        else:
            predict_classes_id = np.argmax(F.softmax(out, dim=1).data.numpy())

        cam_gray = self.makeCAM(feature_data[0], fc_weights, predict_classes_id)
        src_image = cv2.imread(pic_path)
        h, w, _ = src_image.shape
        cam_color_model = cv2.applyColorMap(cv2.resize(cam_gray, (w, h)),
                                               cv2.COLORMAP_HSV)
        cam_model = src_image * 0.5 + cam_color_model * 0.5
        cam_hstack = np.hstack((src_image, cam_model))
        cv2.imwrite("cam_hstack.jpg", cam_hstack)
        Image.open("cam_hstack.jpg").show()


# 模型类
class CNN_model(Tool):
    # 自建模型
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
                # 降维处理
                x = x.view(-1, 1 * 1 * 60)
                x = self.dense(x)
                return x

        # 训练
        cnn = CNN()
        cnn = self.choose_cuda(cnn)

        if self.load:
            return cnn

        lossF = self.choose_lossf()
        optimizer = self.choose_opt()

        # 设置这个可以防止利用gpu时报错
        torch.backends.cudnn.enable = True
        torch.backends.cudnn.benchmark = True
        self.train(cnn, optimizer, lossF)

    # 残差模型
    def model_resent50(self):
        my_resnet50 = models.resnet50(pretrained=True)

        # 由于预训练的模型中的大多数参数已经训练好了，因此先将模型参数的自动梯度求解设为false
        for param in my_resnet50.parameters():
            if self.load:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 将resnet50最后一层输出的类别数，改为ant-bee数据集的类别数，修改后改成梯度计算会恢复为默认的True
        fc_inputs = my_resnet50.fc.in_features
        my_resnet50.fc = nn.Sequential(nn.Linear(fc_inputs, len(self.classes)))
        # 以上操作相当于固定网络全连接层之前的参数，只训练全连接层的参数

        # 损失函数
        lossF = self.choose_lossf()
        # 优化器
        optimizer = self.choose_opt(my_resnet50)
        my_resnet50 = self.choose_cuda(my_resnet50)
        if self.load:
            return my_resnet50
        my_resnet50.train()
        self.train(my_resnet50, optimizer, lossF)

    # alexnet
    def model_alexnet(self):
        my_alexnet = models.alexnet(pretrained=False)
        n_inputs = my_alexnet.classifier[6].in_features  # 4096
        last_layer = nn.Linear(n_inputs, len(self.classes))
        my_alexnet.classifier[6] = last_layer

        # 在alexnet中设置为false会报错
        for param in my_alexnet.parameters():
            param.requires_grad = True

        # 损失函数
        lossF = self.choose_lossf()

        # 优化器
        optimizer = self.choose_opt(my_alexnet)

        my_alexnet = self.choose_cuda(my_alexnet)
        if self.load:
            return my_alexnet

        my_alexnet.train()
        self.train(my_alexnet, optimizer, lossF)

    # lenet 注意lenet需要是一个通道的
    def model_lenet(self):
        class LeNet(nn.Module):
            def __init__(self, num_class=10):
                # num_class为需要分到的类别数
                super().__init__()
                # 输入像素大小为1*28*28
                self.features = nn.Sequential(
                    nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 输出为6*28*28
                    nn.AvgPool2d(kernel_size=2, stride=2),  # 输出为6*14*14，此处也可用MaxPool2d
                    nn.Conv2d(6, 16, kernel_size=5),  # 输出为16*10*10
                    nn.ReLU(),  # 论文中为sigmoid，但极易出现梯度消失
                    nn.AvgPool2d(kernel_size=2, stride=2),  # 输出为16*5*5
                    nn.Flatten()  # 将通道及像素进行合并，方便进一步使用全连接层
                )
                self.classifier = nn.Sequential(
                    nn.Linear(16 * 5 * 5, 120),
                    nn.ReLU(),  # 论文中同样为sigmoid
                    nn.Linear(120, 84),
                    nn.Linear(84, 10))

            def forward(self, x):
                x = self.features(x)
                x = self.classfier(x)

        cnn = LeNet(num_class=len(self.classes))

        # 损失函数
        lossF = self.choose_lossf()

        # 优化器
        optimizer = self.choose_opt(cnn)

        cnn = self.choose_cuda(cnn)
        if self.load:
            return cnn

        cnn.train()
        self.train(cnn, optimizer, lossF)
class CNN_Vir(CNN_model):
    def __init__(self, path, classes, batch_size=32, epochs=20,
                 lr=0.001, device="gpu", opt=0, lossf=0, model_type=0
                 , cifar=False, load=False):
        self.load = load
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
        self.cifar = cifar
        # 加载器与数据集
        self.train_loader, self.test_loader ,\
            self.train_dataset, self.test_dataset = self.pre_data()
        self.model = None
        self.device = device
        self.opt = opt
        self.lossf = lossf
        self.epoch_loss = []


    #预处理数据 返回两个loader 和数据集
    def pre_data(self):
        # 一系列的图片预处理可以由此定义一个对象来处理
        if self.model_type == 0 or self.cifar:
            size = 32
        elif self.model_type == 3:
            size = 28
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
                mean = [0.5, 0.5, 0.5],
                   std = [0.5, 0.5, 0.5]
            )
        ])

        if self.cifar:
            train_dataset = torchvision.datasets.CIFAR10(root=r'./data', train=True, download=True,
                                                      transform=torchvision.transforms.ToTensor())
            test_dataset = torchvision.datasets.CIFAR10(root=r'./data', train=False, download=True,
                                                     transform=torchvision.transforms.ToTensor())
        else:
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

    #绘图
    def draw(self, type="iter", pic_path=None):
        # 展示迭代图
        if type=="iter":
            if self.epoch_loss:
                plt.title("loss vs. Number of Training Epochs")
                plt.xlabel("Training Epochs")
                plt.ylabel("Loss")
                plt.plot(range(self.epochs), self.epoch_loss, label="train")
                plt.xticks(np.arange(0, self.epochs))
                plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
                plt.legend()
                plt.show()
            else:
                raise "Please train first!"
        # 展示原始图片
        elif type=="show_before_train":
            dataiter = iter(self.train_loader)
            images, labels = dataiter.next()
            fig = plt.figure(figsize=(25, 6))
            for idx in np.arange(20):
                ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
                plt.imshow(self.im_convert(images[idx]))
                ax.set_title(self.classes[labels[idx].item()])
            plt.show()
        elif type=="show_after_train":
            if self.device == "gpu":
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            dataiter = iter(self.test_loader)
            images, labels = dataiter.next()
            images = images.to(device)
            labels = labels.to(device)
            output = self.model(images)
            _, preds = torch.max(output, 1)
            fig = plt.figure(figsize=(25, 4))

            for idx in np.arange(20):
                ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
                plt.imshow(self.im_convert(images[idx]))
                ax.set_title("{} ({})".format(str(self.classes[preds[idx].item()]), str(self.classes[labels[idx].item()])),
                             color=("green" if preds[idx] == labels[idx] else "red"))

            plt.show()
        elif type=="cam":
            self.getCam(self.model,pic_path=pic_path)
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
        elif self.model_type == 2:
            self.model = self.model_alexnet()
            self.model.load_state_dict(checkpoint)
        elif self.model_type == 3:
            self.model = self.model_lenet()
            self.model.load_state_dict(checkpoint)

        if self.device == "gpu":
            self.model = self.model.cuda()

    #主程序
    def run(self):
        # 自建模型
        if self.model_type == 0:
            self.cnn_tr()
        # 残差网络
        elif self.model_type == 1:
            self.model_resent50()
        elif self.model_type == 2:
            self.model_alexnet()
        elif self.model_type == 3:
            self.model_lenet()

#定义数据集路径
path = "./SDNET2018/light"
#车辆种类
classes = ["cracked", "non-cracked"]
classes_cifar = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# test = CNN_Vir(path, classes, device="gpu", model_type=2, epochs=200, batch_size=64, lossf=0, opt=3, lr=0.1)
# test = CNN_Vir(path, classes_cifar, device="gpu", model_type=1, epochs=10, batch_size=128, lossf=0, opt=3, lr=0.1, cifar=True)
# test = CNN_Vir(path, classes, device="gpu", model_type=2, epochs=25, batch_size=32, lossf=0, opt=3, lr=0.1, load=True)
# test.load_model(name="alexnet_xxx.pth")
# print(test.model)
# test.draw(type="cam", pic_path="001-104.jpg")
# test.draw(type="cam", pic_path="cam_test.jpg")




class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al.
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''

    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers

        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)

        self.activations = []
        self.grads = []

    def forward_hook(self, module, input, output):
        self.activations.append(output[0])

    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())

    def calculate_cam(self, model_input):
        if self.use_cuda:
            device = torch.device('cuda')
            self.model.to(device)  # Module.to() is in-place method
            model_input = model_input.to(device)  # Tensor.to() is not a in-place method
        self.model.eval()

        # forward
        y_hat = self.model(model_input)
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)

        # backward
        self.model.zero_grad()
        y_c = y_hat[0, max_class]
        y_c.backward()

        # get activations and gradients
        activations = self.activations[0].cpu().data.numpy().squeeze()
        grads = self.grads[0].cpu().data.numpy().squeeze()

        # calculate weights
        weights = np.mean(grads.reshape(grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1, 1)
        cam = (weights * activations).sum(axis=0)
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max()
        return cam

    @staticmethod
    def show_cam_on_image(image, cam):
        # image: [H,W,C]
        h, w = image.shape[:2]

        cam = cv2.resize(cam, (h, w))
        cam = cam / cam.max()
        heatmap = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)  # [H,W,C]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        image = image / image.max()
        heatmap = heatmap / heatmap.max()

        result = 0.4 * heatmap + 0.6 * image
        result = result / result.max()

        plt.figure()
        plt.imshow((result * 255).astype(np.uint8))
        plt.colorbar(shrink=0.8)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        preprocessing = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        return preprocessing(img.copy()).unsqueeze(0)



alexnet_standard = CNN_Vir(path, classes, device="gpu", model_type=1, epochs=1, batch_size=32, lossf=0, opt=3, lr=0.1, load=True)
alexnet_standard.load_model("restnet_xxx.pth")
for i in ['1.jpg', '2.jpg', '3.jpg']:
    image = cv2.imread(f'./grad_cam/{i}')  # (224,224,3)
    input_tensor = GradCAM.preprocess_image(image)

    grad_cam = GradCAM(alexnet_standard.model, alexnet_standard.model.layer4[-1], 224)
    cam = grad_cam.calculate_cam(input_tensor)
    GradCAM.show_cam_on_image(image, cam)


