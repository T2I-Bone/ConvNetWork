import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from para import cfg


def train_neuralNetwork(trainDataset, network,
                        netName, modelPath='./model/', load_para=False):
    train_dataloader = torch.utils.data.DataLoader(
        trainDataset, batch_size=cfg.batch_size,
        drop_last=True, shuffle=cfg.bshuffle, num_workers=int(cfg.workers))
    num_batches = len(train_dataloader)
    if load_para and os.path.exists(modelPath + netName + '.pth'):
        print('load neural network from : ' + modelPath + netName + '.pth')
        network = torch.load(modelPath + netName + '.pth')
    network.train()
    if cfg.CUDA:
        network = network.cuda(device=cfg.device)
    opt = cfg.optimizer(network.parameters(),
                        lr=cfg.lr,
                        betas=(0.5, 0.999))
    loss = cfg.loss
    max_epoch = cfg.num_epoch
    Loss = []
    Acc = []
    for epoch in range(0, max_epoch):
        data_iter = iter(train_dataloader)
        loss_epoch = 0
        for step_num in range(0, num_batches):
            data = data_iter.next()
            imgs, labels = data
            imgs = Variable(imgs.float()).cuda(device=cfg.device)
            labels = Variable(labels.float()).cuda(device=cfg.device)
            network.zero_grad()
            pred = network(imgs)
            loss_step = loss(pred, labels)
            loss_epoch += loss_step
            loss_step.backward()
            opt.step()
            if (epoch * num_batches + step_num) % cfg.interval_sample == 0:
                print('save model to ' + modelPath + netName + '.pth')
                torch.save(network, modelPath + netName + '.pth')
                acc_tmp = Sample_Accuracy(network, trainDataset)
                print(f'epoch: {epoch} , Loss : {loss_step.data:.5f} , accuracy: {acc_tmp:.5f}')

        Loss.append(loss_epoch)
        Acc.append(Sample_Accuracy(network, trainDataset))
    drawAccuracyAndLoss(Loss, Acc)


def Sample_Accuracy(network, trainDataset):
    N = len(trainDataset)
    data_features = trainDataset.data_features[99 * N // 100:N]
    data_labels = trainDataset.data_label[99 * N // 100:N]
    data_features_tensor = Variable(torch.tensor(data_features, dtype=torch.float32)).cpu()
    pred_sample = (network.cpu())(data_features_tensor)

    label_pred = np.array(torch.argmax(pred_sample, axis=1))
    network.cuda(device=cfg.device)
    accuracy = np.mean(label_pred == data_labels)
    return accuracy


def drawAccuracyAndLoss(Loss, Acc, pathname='./output'):
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    Loss_arr = np.array(Loss)
    Acc_arr = np.array(Acc)
    epoch_arr = np.arange(Loss_arr)
    plt.figure(figsize=(10, 10))
    plt.plot(epoch_arr, Loss_arr, label='Loss')
    plt.plot(epoch_arr, Acc_arr, label='Accuracy')
    plt.title('Loss and Accuracy')
    plt.legend()
    plt.savefig(pathname + '/LossAcc.png')
