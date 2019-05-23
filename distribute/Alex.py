# coding: utf-8
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def main():
    import torch
    from torch import optim
    from optimizer.lars import LARS
    data = torch.randn((2, 3, 227, 227))
    label = torch.Tensor([0, 1]).long()
    net = AlexNet(10)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    optimizer1 = LARS(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criteral = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    optimizer1.zero_grad()

    out = net(data)
    loss = criteral(out, label)
    loss.backward()

    optimizer1.step()
    optimizer.step()


if __name__ == '__main__':
     main()
    # import torch
    # a = torch.Tensor([0, 1])
    # b = torch.Tensor([1, 2])
    # c = torch.Tensor([3, 4])
    # a.add_(b, c)
    # print ( 123)