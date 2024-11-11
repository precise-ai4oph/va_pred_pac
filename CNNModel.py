import torch
import torch.nn.functional as F

class LeNet5(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = torch.nn.Conv2d(6,16,5)
        self.lin1 = torch.nn.Linear(250000, 120)
        self.lin2 = torch.nn.Linear(120, 84)
        self.lin3 = torch.nn.Linear(84, n_classes)


    def forward(self, x):
        x = self.feature(x)
        x = self.lin3(x)
        return x

    def feature(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x 

class SimpleCNN(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(SimpleCNN, self).__init__()
        print("N_CLASSES", n_classes)

        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, stride=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 5, stride=1)
        self.lin1 = torch.nn.Linear(57600, 64)
        self.lin2 = torch.nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

