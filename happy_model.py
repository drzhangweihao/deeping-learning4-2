import  torch
import torch.nn as nn
from torch.autograd import Variable
import  torch.nn.functional as F
from kt_utils import load_dataset
import torch.utils.data as Data
import  numpy as np
import matplotlib.pyplot as plt
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
model.load_state_dict(torch.load('model_happy.pkl'))
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
test_x_plot = X_test_orig.copy()
test_x = X_test_orig/255
test_y = Y_test_orig.T
test_x = test_x.reshape((test_x.shape[0],test_x.shape[3],test_x.shape[1],test_x.shape[2]))
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()
test_out = model(test_x)
test_prediction = (test_out.data.cpu().numpy() >= 0.5)
a = {1:'happy',0:'unhappy'}

index = np.random.randint(0,120)
plt.imshow(test_x_plot[index])
plt.title(a[(test_prediction)[index][0]])
plt.show()
