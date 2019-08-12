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


def load_model(path_):
    model = torch.load(path_)
    return model


def prepare_data(**kwargs):
    data_transf = transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.ToTensor()])
    submit = pd.read_csv(path + 'sample_submission.csv', converters={'EncodedPixels': lambda e: ' '})
    test_data = ImageData(df=submit, transform=data_transf, subset="test")
    test_loader = DataLoader(dataset=test_data, shuffle=False)
    kwargs['model_vals'].xcom_push(key='test_loader', value=test_loader)


def predict(**kwargs):
    model_vals = kwargs['model_vals']
    model = load_model('model.pth')
    model.eval()
    predict = []
    test_loader = model_vals.xcom_pull(key='test_loader', task_ids='prepare_data')
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
    kwargs['model_vals'].xcom_push(key='predict_arr', value=predict)


def postprocess(**kwargs):
    model_vals = kwargs['model_vals']
    predict = model_vals.xcom_pull(key='predict_arr', task_ids='predict')
    submit = pd.read_csv(path + 'sample_submission.csv', converters={'EncodedPixels': lambda e: ' '})
    submit['EncodedPixels'] = predict
    submit.to_csv('submission.csv', index=False)
