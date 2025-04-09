import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_features, num_targets):
        super(CNN, self).__init__()
        # conv1: 使用 padding=1 保持序列长度
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # conv2: 使用 padding=1 保持序列长度
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=320, kernel_size=3, padding=1)
        # conv3: 将通道数调整为 num_targets (5)，kernel_size=1 不改变长度
        self.conv3 = nn.Conv1d(in_channels=320, out_channels=num_targets, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)    # (1, 5, 320) -> (1, 64, 320)
        x = self.relu(x)
        x = self.conv2(x)    # (1, 64, 320) -> (1, 320, 320)
        x = self.relu(x)
        x = self.conv3(x)    # (1, 320, 320) -> (1, 5, 320)
        return x

if __name__ == '__main__':
    input = torch.randn(1, 5, 320)
    cnn = CNN(num_features=5, num_targets=5)
    output = cnn(input)
    print(output.shape)  # 输出: torch.Size([1, 5, 320])