# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm_notebook
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms
import torch.nn.functional as F
from utils import ImageData, rle2mask, mask2rle, prepare_data, make_aug
from unet import UNet


path = 'data/'


def train_model():
    model = UNet(n_class=1).cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, lr=0.001, momentum=0.9)
    train_loader, test_loader = prepare_data()
    for epoch in range(5):
        model.train()
        for ii, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))
    torch.save(model.state_dict(), 'my_model.pth')
    return model, test_loader


def make_predict(model):
    model.eval()
    predict = []
    for data in test_loader:
        data = data.cuda()
        output = model(data)
        output = output.cpu().detach().numpy() * (-1)
        img = np.copy(abs(output[0]))
        mn = np.mean(img) * 1.2
        img[img <= mn] = 0
        img[img > mn] = 1
        img = cv2.resize(img[0], (1600, 256))
        predict.append(mask2rle(img))
    return predict


if __name__ == '__main__':
    submit = pd.read_csv(path + 'sample_submission.csv', converters={'EncodedPixels': lambda e: ' '})
    model, test_loader = train_model()
    predict = make_predict(model)
    submit['EncodedPixels'][submit['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4')] = predict
    submit.to_csv('submission.csv', index=False)
