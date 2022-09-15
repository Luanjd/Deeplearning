import torch
import torch.nn as nn


class Bottleneck(nn.Module):

    extention = 2

    def __init__(self, in_ch, out_ch, stride, side):
        super(Bottleneck, self).__init__()

        # Stage中具体的网络结构
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

        self.side = side

        
    def forward(self, x):

        # 残差边的初始输入
        residual = x

        # Stage中的两个Bottleneck
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))

        # 残差边的输入x是否经过处理
        if self.side != None:
            residual = self.side(x)

        out += residual
        out = self.relu(out)

        return out



class Resnet(nn.Module):

    def __init__(self, block, num_classes=1000):

        self.block = block
        # Stage1最初输入输出通道数
        self.in_ch = 64
        self.out_ch = 64

        super(Resnet, self).__init__()

        # stage0的网络
        self.conv1 = nn.Conv2d(3, self.out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_ch)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 中间4个Stage网络
        self.stage1 = self.make_layer(self.block, self.in_ch, stride=1)
        self.stage2 = self.make_layer(self.block, self.in_ch, stride=2)
        self.stage3 = self.make_layer(self.block, self.in_ch, stride=2)
        self.stage4 = self.make_layer(self.block, self.in_ch, stride=2)

        # 后续分类处理网络
        self.avgpool = nn.AvgPool2d(7)  # 7???
        self.fc = nn.Linear(self.in_ch, num_classes)


    def make_layer(self, block, in_ch, stride):

        block_list = []

        # 判断是Identity Block还是Conv Block，由此来决定残差边处理值
        if stride != 1:
            side = nn.Sequential(
                nn.Conv2d(in_ch, self.out_ch, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(self.out_ch))
        else:
            side = None

        block1 = block(in_ch, self.out_ch, stride=stride, side=side)
        block_list.append(block1)
        block2 = block(self.out_ch, self.out_ch, stride=1, side=None)
        block_list.append(block2)

        self.in_ch = self.out_ch
        self.out_ch = self.in_ch * block.extention

        return nn.Sequential(*block_list)


    def forward(self, x):

        # stage0的实现
        out = self.maxpooling(self.relu(self.bn1(self.conv1(x))))

        # block
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # 分类
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    
    
# 调用
resnet = Resnet(Bottleneck)
x = torch.rand(1, 3, 224, 224)
x = resnet(x)
print(x.shape)
