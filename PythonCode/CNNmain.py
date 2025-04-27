## 导入包
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchinfo import summary
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay
# 构建网络
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1),
            nn.Flatten(),
            nn.Linear(in_features = 32*110*110, out_features = 2),
            )
    def forward(self,x):
        x = self.layer(x)
        return x
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnnnet = CNNnet().to(device)

# 可视化一下
summary(cnnnet, (10, 3, 224, 224)) # (批大小，图像通道，图像长，图像宽)

# 定义图像转换函数
dataTrans = transforms.Compose([
    transforms.Resize([224,224]), # 图像尺寸统一为224*224
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 这个参数参考网上，可能是个经验值,一般的图像分类就这样用就行
    ]) 

# 加载数据集
data_dir = r'CatDogs/'

train_dir = data_dir + 'train';
test_dir = data_dir + 'test';

trainset = datasets.ImageFolder(train_dir, dataTrans) # 加载图像训练集
testset = datasets.ImageFolder(test_dir, dataTrans) # 加载图像测试集
print(trainset.class_to_idx)

print('训练集数目：', len(trainset))
print('测试集数目：', len(testset))

# 用DataLoader打包，如果想一批一批训练，这个是必须的
train_dataloader = DataLoader(trainset, batch_size = 15, shuffle = True)
test_dataloader = DataLoader(testset, batch_size = len(testset), shuffle = False)
# 查看一批batch的数据格式
x_train, y_train = next(iter(train_dataloader))
print(x_train.shape, y_train.shape)

# 训练
# 定义优化器和损失
optimizer = torch.optim.SGD(cnnnet.parameters(),lr = 0.01)
loss_func = nn.CrossEntropyLoss()
# 开始训练
train_loss = 0

for epoch in range (10):
    for i,(x_train,y_train) in enumerate(train_dataloader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        outputs = cnnnet(x_train).to(device)
        # 计算误差
        loss = loss_func(outputs, y_train.long())
        # 计算准确率
        _, y_pred = torch.max(outputs.data, dim=1)
        # 清空上一次梯度
        optimizer.zero_grad()
        # 反向传播、更新参数
        loss.backward()
        optimizer.step()
        # 计算训练集acc、loss并输出
        train_acc = (y_pred == y_train).sum() / len(y_train)
        train_loss += loss.item()
        if (i+1) % 10 == 0: # 每10次迭代输出一次
            print('[%d %5d] loss: %.3f acc: %.3f' % (epoch + 1, i + 1, train_loss / 10, train_acc))
            train_loss = 0.0

# 测试
correct = 0
with torch.no_grad():
    for x_test, y_test in test_dataloader: # 这个只循环一次，test_dataloader里就是全部测试集样本，没有分批次
        x_test, y_test = x_test.to(device), y_test.to(device)
        outputs = cnnnet(x_test)
        _, y_pred = torch.max(outputs.data, dim=1)
        correct += (y_pred == y_test).sum()
print('Accuracy of the test images: %.3f %%' % (100 * correct / len(x_test)))

# 计算指标
y_test, y_pred = y_test.cpu(), y_pred.cpu()
acc = accuracy_score(y_test,y_pred) 
pre = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred) 
f1score = f1_score(y_test,y_pred) 
print('计算指标结果：\nAcc: %.2f%% \nPre: %.2f%% \nRecall: %.2f%% \nF1-score: %.2f%% ' % (100*acc,100*pre,100*recall,100*f1score))

## 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()

## 保存模型
example_forward_input = torch.rand(1, 3, 224, 224)
cnnnet.to("cpu")
module = torch.jit.trace(cnnnet.forward, example_forward_input)
torch.jit.save(module, "cnnnet.pt")