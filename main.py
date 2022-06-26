import os.path

import numpy as np
import pandas as pd
import torch
from Networks import getConvNetwork
from dataset import MNISTDataset

# 64*1*28*28->64*10
from para import cfg
from testNetwork import test_neuralNetwork, drawSingleDigit
from trainNetwork import train_neuralNetwork


def getData(filename):
    Data_df = pd.read_csv(filename)
    Data_arr = np.array(Data_df)
    Data_tor = torch.Tensor(Data_arr)
    print(Data_tor.shape)


if __name__ == '__main__':
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./output'):
        os.mkdir('./output')

    # ===================训练=================
    trainDataset = MNISTDataset(cfg.trainFilename, train=True)
    CNN = getConvNetwork(cfg.networkName,
                         cfg.vgg_conv_inf, cfg.vgg_fc_in, cfg.vgg_fc_hidden, cfg.vgg_fc_out,
                         cfg.resnet_in_channels)
    train_neuralNetwork(trainDataset, CNN, netName=cfg.networkName, load_para=cfg.load_para,)
    trainDataset.labelCount()
    trainDataset.imageshow()

    # ==================预测==================
    testDataset = MNISTDataset(cfg.testFilename, train=False)
    predict_label = test_neuralNetwork(testDataset, cfg.networkName)
    # print(predict_label.iloc[:10, :])
    # for i in range(0, 10):
    #     drawSingleDigit(testDataset, i)

    # img = torch.ones((128, 1, 28, 28))
    # googleNet = getConvNetwork('GoogLet')
    # print(googleNet)
    # resnet = ResNet(in_channels=img.shape[1])
    # print(le_net(img).shape)
    # print(alex_net(img).shape)
    # print(vgg_net(img).shape)
    # print(google_net(img).shape)
    # print(resnet(img).shape)
    # blk=Residual(3,6,use_1x1conv=True,stride=1)
    # X=torch.rand((4,3,6,6))
    # print(blk(X).shape)
    # # print(blk)
    # print(resnet)

    # X = torch.rand((1, 1, 224, 224))
    # for name, layer in resnet.named_children():
    #     X = layer(X)
    #     print(name, ' output shape:\t', X.shape)
