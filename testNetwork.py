import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from para import cfg


def drawSingleDigit(dataSet, index):
    plt.figure(figsize=(8, 8))
    img = (dataSet.data_f_arr[index]).reshape(28, 28)
    plt.imshow(img)
    plt.show()


def test_neuralNetwork(trainDataset, netWorkName,
                       modelPath='./model/'):
    try:
        print('load neural network from : ' + modelPath + netWorkName + '.pth')
        network = torch.load(modelPath + netWorkName + '.pth')
    except:
        print('the model pth file does not exist')
    data_features = trainDataset.data_features
    data_features_tensor = torch.tensor(data_features, dtype=torch.float32)
    pred_sample = (network.cpu())(data_features_tensor)
    network.cuda(device=cfg.device)
    label_pred = np.array(torch.argmax(pred_sample, axis=1))
    label_df = pd.DataFrame(
        label_pred, index=range(1, len(label_pred) + 1), columns=['label']
    )
    label_df.insert(loc=0, column='id', value=range(0, len(label_pred)))
    label_df.to_csv('./output/predict.csv', index=False)
    return label_df
