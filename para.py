import torch
from easydict import EasyDict as edict

cfg = edict()
cfg.trainFilename = './data/train.csv'
cfg.testFilename = './data/test.csv'
cfg.networkName = 'AlexNet'  # LeNet  AlexNet  VGGNet   GoogLet  ResNet
cfg.batch_size = 256
cfg.load_para = False
cfg.lr = 0.001
cfg.num_epoch = 30
cfg.bshuffle = True
cfg.workers = 4
cfg.train_flag = False
cfg.interval_sample = 100
cfg.CUDA = True
cfg.loss = torch.nn.CrossEntropyLoss()
cfg.optimizer = torch.optim.Adam
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.vgg_conv_inf = ((1, 1, 64), (1, 64, 128), (2, 128, 256))
cfg.vgg_fc_in = 256
cfg.vgg_fc_hidden = 64
cfg.vgg_fc_out = 10
cfg.resnet_in_channels = 1
