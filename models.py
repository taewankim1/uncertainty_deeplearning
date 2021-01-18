import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class LinearNN(nn.Module):
    def __init__(self,num_hidden,drop_prob):
        # super(LinearNN, self).__init__()
        super().__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden) 
        self.fc3 = nn.Linear(num_hidden, 1)
        self.dropout1 = torch.nn.Dropout(p=drop_prob)
        self.dropout2 = torch.nn.Dropout(p=drop_prob)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_uniform_(m.weight)
                # m.bias.data.zero_()
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class LinearDensityNN(nn.Module):
    def __init__(self,num_hidden,drop_prob):
        # super(LinearNN, self).__init__()
        super().__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden) 
        self.fc3_1 = nn.Linear(num_hidden, 1)
        self.fc3_2 = nn.Linear(num_hidden, 1)
        self.dropout1 = torch.nn.Dropout(p=drop_prob)
        self.dropout2 = torch.nn.Dropout(p=drop_prob)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_uniform_(m.weight)
                # m.bias.data.zero_()
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        mu = self.fc3_1(x)
        sigma = torch.exp(self.fc3_2(x))
        return mu, sigma