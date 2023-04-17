import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    # Defining the Constructor
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=24, out_channels=36, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=36, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)

        # 128x128 image tensors will be pooled twice
        self.fc = nn.Linear(in_features=8 * 8 * 64, out_features=36)
        # self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=36, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))

        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))

        x = F.dropout(self.drop(x), training=self.training)

        x = x.view(-1, 8 * 8 * 64)

        x = F.relu(self.fc(x))
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Return class probabilities via a log_softmax function
        return F.log_softmax(x, dim=1)
