import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self,pts):
        super().__init__()
        self.fc1 = nn.Linear(pts,128) #fully connected (fc) layer
        self.fc2 = nn.Linear(128, 64)  # fully connected (fc) layer
        self.fc3 = nn.Linear(64, 32)  # fully connected (fc) layer
        self.fc4 = nn.Linear(32, 16)  # 2= number of outputs
        self.fc5 = nn.Linear(16, 8)  # fully connected (fc) layer
        self.fc6 = nn.Linear(8, 4)  # fully connected (fc) layer
        self.fc7 = nn.Linear(4, 2)  # fully connected (fc) layer

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        # return F.linear(x,torch.Tensor([1,1]).view((1,2)))
        return F.log_softmax(x,dim=-1) #probability distrib on output