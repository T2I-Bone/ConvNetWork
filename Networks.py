import torch
from torch import nn
import torch.nn.functional as F


def getConvNetwork(networkName,
                   vgg_conv_inf=None, vgg_fc_in=None, vgg_fc_hidden=None, vgg_fc_out=None,
                   resnet_in_channels=None):
    if networkName == 'LeNet':
        return LeNet()
    elif networkName == 'AlexNet':
        return AlexNet()
    elif networkName == 'VGGNet':
        vgg_net = VGGNet(conv_inf=vgg_conv_inf,
                         fc_in=vgg_fc_in, fc_hidden=vgg_fc_hidden, fc_out=vgg_fc_out)
        return vgg_net
    elif networkName == 'GoogLet':
        return GoogLeNet()
    elif networkName == 'ResNet':
        return ResNet(resnet_in_channels)


# LeNet , AlexNet , VGGNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=1, stride=1),
            # [1,28,28]->[6,28,28]
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [6,28,28]->[6,14,14]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            # [6,14,14]->[16,10,10]
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # [16,10,10]->[16,5,5]
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            # [batchSize,16,5,5]->[batchSize,16*5*5]
            # ignore shape[0] [16*5*5]->[120]
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),
            # ignore shape[0] [120]->[84]
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
            # ignore shape[0] [84]->[10]
        )
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, img):
        feature = self.conv(img)
        out = self.fc(feature.view(img.shape[0], -1))
        # [b,16,5,5]->[b,16*5*5]
        return out


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            # [1,28,28]->[32,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32,28,28]->[32,14,14]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # [32,14,14]->[64,14,14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [64,14,14]->[64,7,7]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # [64,7,7]->[128,7,7]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # [128,7,7]->[256,7,7]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # [256,7,7]->[256,7,7]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # [256,7,7]->[256,3,3]
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=256 * 3 * 3, out_features=1024),
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=10)
        )
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, img):
        feature = self.conv(img)
        out = self.fc(feature.view(feature.shape[0], -1))
        return out


class VGGNet(nn.Module):
    def __init__(self, conv_inf, fc_in, fc_hidden, fc_out):
        super(VGGNet, self).__init__()
        self.conv = nn.Sequential()
        # ((1,1,64),(1,64,128),(2,128,256))
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_inf):
            block_tmp = self.vgg_block(num_convs, in_channels, out_channels)
            self.conv.add_module("vgg_block_" + str(i + 1), block_tmp)
        # [1,28,28]->[64,14,14]->[128,7,7]->[256,3,3]->[256,1,1]
        self.fc = nn.Sequential()
        # [b,256,1,1]->[b,256]->[b,64]->[b,64]->[b,10]
        self.fc.add_module("fc", nn.Sequential(nn.Linear(in_features=fc_in, out_features=fc_hidden),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(p=0.5),
                                               nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(p=0.5),
                                               nn.Linear(in_features=fc_hidden, out_features=fc_out)
                                               ))

        self.weight_init()

    def vgg_block(self, num_conv, in_channels, out_channels):
        """
        :param num_conv:  卷积层的数目
        :param in_channels:  输入channel
        :param out_channels: 输出 channel
        :return: nn.Sequential : 一个 VGG block
        """
        block = []
        for i in range(num_conv):
            if i == 0:
                block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
                # [in,a,a]->[out,a,a]
            else:
                block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
                # [out,a,a]->[out,a,a]
            block.append(nn.ReLU(inplace=True))
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # [C,a,a]->[C,a//2,a//2]
        return nn.Sequential(*block)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, img):
        features = self.conv(img)
        out = self.fc(features.view(img.shape[0], -1))
        return out


# Inception Block
class Inception(nn.Module):
    def __init__(self, in_c, c1_out, c2_out, c3_out, c4_out):
        super(Inception, self).__init__()
        """
            大小不发生改变，仅仅是通道数发生改变 1x1 3x3 5x5 (maxPool)1x1
        """
        self.net_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=c1_out, kernel_size=1, stride=1),
            # [in_c,a,a]->[c1_out,a,a]
            nn.ReLU()
        )
        self.net_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=c2_out[0], kernel_size=1, stride=1),
            nn.ReLU(),
            # [in_c,a,a]->[c1_out,a,a]
            nn.Conv2d(in_channels=c2_out[0], out_channels=c2_out[1], kernel_size=3, stride=1, padding=1),
            # [c2_out[0],a,a]->[c2_out[1],a,a]
            nn.ReLU()
        )
        self.net_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=c3_out[0], kernel_size=1, stride=1),
            # [in_c,a,a]->[c3_out[0],a,a]
            nn.ReLU(),
            nn.Conv2d(in_channels=c3_out[0], out_channels=c3_out[1], kernel_size=5, stride=1, padding=2),
            # [c3_out[0],a,a]->[c3_out[1],a,a]
            nn.ReLU()
        )
        self.net_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # [in_c,a,a]->[in_c,a,a]
            nn.Conv2d(in_channels=in_c, out_channels=c4_out, kernel_size=1, stride=1),
            # [in_c,a,a]->[c4_out,a,a]
            nn.ReLU()
        )

    def forward(self, x):
        n1 = self.net_1(x)
        n2 = self.net_2(x)
        n3 = self.net_3(x)
        n4 = self.net_4(x)

        out = torch.cat((n1, n2, n3, n4), dim=1)
        return out


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3),
            # [1,a,a]->[64,a,a]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # [64,a,a]->[64,a//2,a//2]  # 28->14
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            # [64,a,a]->[64,a,a]
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            # [64,a,a]->[192,a,a]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # [192,a,a]->[192,a//2,a//2]  #14->7
        )
        self.block3 = nn.Sequential(
            Inception(in_c=192, c1_out=64, c2_out=(96, 128), c3_out=(16, 32), c4_out=32),
            # [192,a,a]->[64+128+32+32,a,a]=[256,a,a]
            Inception(in_c=256, c1_out=128, c2_out=(128, 192), c3_out=(32, 96), c4_out=64),
            # [256,a,a]->[128+192+96+64,a,a]=[480,a,a]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # [480,a,a]->[480,a//2,a//2] # 7->3
        )
        self.block4 = nn.Sequential(
            Inception(in_c=480, c1_out=192, c2_out=(96, 208), c3_out=(16, 48), c4_out=64),
            # [480,a,a]->[192+208+48+64,a,a]=[512,a,a]
            Inception(in_c=512, c1_out=160, c2_out=(112, 224), c3_out=(24, 64), c4_out=64),
            # [512,a,a]->[160+224+64+64,a,a]=[512,a,a]
            Inception(in_c=512, c1_out=128, c2_out=(128, 256), c3_out=(24, 64), c4_out=64),
            # [512,a,a]->[128+256+64+64,a,a]=[512,a,a]
            Inception(in_c=512, c1_out=112, c2_out=(144, 288), c3_out=(32, 64), c4_out=64),
            # [512,a,a]->[112+288+64+64,a,a]=[528,a,a]
            Inception(in_c=528, c1_out=256, c2_out=(160, 320), c3_out=(32, 128), c4_out=128),
            # [528,a,a]->[256+320+128+128,a,a]=[832,a,a]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # [832,a,a]->[832,a//2,a//2]
        )
        self.block5 = nn.Sequential(
            Inception(in_c=832, c1_out=256, c2_out=(160, 320), c3_out=(32, 128), c4_out=128),
            # [832,a,a]->[256+320+128+128,a,a]=[832,a,a]
            Inception(in_c=832, c1_out=384, c2_out=(48, 128), c3_out=(32, 128), c4_out=128),
            # [832,a,a]->[384+128+128+128,a,a]=[768,a,a]
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 10)
        )
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, img):
        f1 = self.block1(img)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        f5 = self.block5(f4)
        return self.fc(f5)


# ResNet
class Residual(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 use_1x1conv=False,
                 stride=1):
        super(Residual, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        if use_1x1conv:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, X):
        Y = self.conv_base(X)
        if self.conv1x1:
            X = self.conv1x1(X)
        return F.relu(Y + X)


class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.net = nn.Sequential(
            self.resnet_block(64, 64, 2, first_block=True),
            self.resnet_block(64, 128, 2),
            self.resnet_block(128, 256, 2),
            self.resnet_block(256, 512, 2)
            # [128, 512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )
        self.weight_init()

    def resnet_block(self, in_channels, out_channels,
                     num_residuals, first_block=False):
        if first_block:
            assert in_channels == out_channels
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, img):
        out = self.conv(img)
        out = self.net(out)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        return self.fc(out)
