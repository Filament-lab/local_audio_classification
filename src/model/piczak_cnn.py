from torch import nn
from torch.functional import F


class PiczakCnn(nn.Module):
    def __init__(self, num_classes: int):
        """
        Initialize layers
        :param num_classes: Number of classes for output layer
        """
        super(PiczakCnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=80, kernel_size=(57, 6), stride=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3), stride=(1, 1), padding=0)
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=8240, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.3, inplace=False)
        self.num_classes = num_classes

    def forward(self, x):
        # Conv layer 1.
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(4, 3))

        # Conv layer 2.
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(1, 3))

        # Flatten
        x = self.flatten1(x)

        # Fully connected layer 1
        x = self.fc1(x)
        x = F.relu(x)

        # Fully connected layer 2
        x = self.fc2(x)

        # Softmax
        x = F.log_softmax(x)

        return x
