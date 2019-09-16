import  torch
import torch.nn as nn
from torch.autograd import Variable
import  torch.nn.functional as F
from kt_utils import load_dataset
import torch.utils.data as Data
import  numpy as np
import matplotlib.pyplot as plt

#加载数据
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#初始化数据
train_x = X_train_orig/255
test_x = X_test_orig/255
train_y = Y_train_orig.T
test_y = Y_test_orig.T
Batch_size = 32
LR = 0.005
EPOCH = 500
cost = []
cost2 =[]

# 数据维度
print('train_x.shape:',train_x.shape)
print('test_x.shape:',test_x.shape)

print('train_y.shape:',train_y.shape)
print('test_y.shape:',test_y.shape)

#加载数据 转化类型
train_x = train_x.reshape((train_x.shape[0],train_x.shape[3],train_x.shape[1],train_x.shape[2]))
test_x = test_x.reshape((test_x.shape[0],test_x.shape[3],test_x.shape[1],test_x.shape[2]))
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()
train_data = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(dataset= train_data,batch_size=Batch_size,shuffle=True)


class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=(7,7),stride=1,padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=1,padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.out = nn.Linear(8*8*32,1)

    def forward(self, x):
        x = self.conv1(x)      #(8,32,32)
        x = self.conv2(x)      #(16,16,16)
        x = self.conv3(x)      #(32,8,8)
        x = x.view(x.size(0),-1)   #(64,2048)
        x = self.out(x)         #(64,1)
        out = F.sigmoid(x)
        return out

model = cnn()

if torch.cuda.is_available():
    model = cnn().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
loss_func = nn.MSELoss()
for epoch in range(EPOCH):
    for step ,(x,y) in enumerate(train_loader):
        if torch.cuda.is_available():
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
        else:
            b_x = Variable(x)
            b_y = Variable(y)
        out = model(b_x)
        loss = loss_func(out,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            if step % 5 == 0:
                pred_train_y = (out.data.cpu().numpy() >= 0.5)
                accauracy_train = np.sum(pred_train_y == b_y.data.cpu().numpy()) /b_y.size(0)
                print('EPOCH:',epoch,'step:',step,'loss: %.4f' %loss.data,'accauracy:%.4f' %accauracy_train)
                cost.append(loss.data)
                test_out = model(Variable(test_x).cuda())
                test_prediction = (test_out.data.cpu().numpy() >= 0.5)
                test_accauracy = np.sum(test_prediction == test_y.data.cpu().numpy()) / test_y.size(0)
                print('accauracy_test:%.4f' % test_accauracy)
                cost2.append(test_accauracy)

plt.subplot(1,2,1)
plt.plot((cost))
plt.title('lr = 0.001')
plt.ylabel('cost')
plt.xlabel('iterations(pertens)')

plt.subplot(1,2,2)
plt.plot((cost2))
plt.title('lr = 0.001')
plt.ylabel('accauracy')
plt.xlabel('iterations(pertens)')
plt.show()

# model = model.cpu()
# test_out = model(test_x)
# test_prediction = (test_out.data.numpy() >= 0.5)
# test_accauracy = np.sum(test_prediction == test_y.numpy()) / test_y.size(0)
# print('accauracy_test:%.4f' % test_accauracy)
# cost2.append(test_accauracy)

torch.save(model.state_dict(),'model_happy.pkl')