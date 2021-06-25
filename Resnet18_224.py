import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

traindataset = datasets.ImageFolder(root='./data/train',
                                    transform=data_transform)
trainloader = DataLoader(dataset=traindataset,
                         batch_size=6,
                         shuffle=True,
                         num_workers=0)

testdataset = datasets.ImageFolder(root='./data/test',
                                   transform=data_transform)
testloader = DataLoader(dataset=testdataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=0)

dataiter = iter(trainloader)
images, labels = dataiter.next()
images.shape
torchvision.utils.save_image(images[1], "test_Resnet18.jpg")


def imshow(inp, title, ylabel):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()
    plt.ylabel('GroundTruth: {}'.format(ylabel))
    plt.title('predicted: {}'.format(title))


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=2):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)


net = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

"""
for epoch in range(40):
    train_loss = 0.0
    for batch_idx, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % 20 == 19:
            print('[%d,%5d]loss:%.3f' % (epoch + 1, batch_idx + 1, train_loss ))
            train_loss = 0.0
    print('Saving epoch%d model ......' % (epoch + 1))
    state = {
                'net': net.state_dict(),
                'epoch': epoch + 1,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/Resnet18_epoch_%d.ckpt' % (epoch + 1))

print('Finished Training :)')
"""

# 加载训练好的模型
checkpoint = torch.load('./checkpoint/Resnet18_epoch_8.ckpt')
net.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']

dataiter = iter(testloader)
test_images, test_labels = dataiter.next()

outputs = net(test_images)
_, predicted = torch.max(outputs, 1)

classes = testdataset.classes
i = 1
j = 0
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        out = torchvision.utils.make_grid(images)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        print(i, '.Predicted:', ''.join('%5s' % classes[predicted]), '  GroundTruth:', ''.join('%5s' % classes[labels]))
        if j % 4 == 0:
            plt.figure()
            j = j % 4
        plt.subplot(2, 2, j + 1)
        imshow(out, title=[classes[predicted]], ylabel=[classes[labels]])
        j = j + 1
        i = i + 1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


