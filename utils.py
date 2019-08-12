# coding: utf-8
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import cv2
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma
)


path = 'data/'


def prepare_data():
    tr = pd.read_csv(path + '/train/train.csv')
    df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)
    df_aug = make_aug(df_train)
    df_train.append(df_aug, ignore_index=True)
    data_transf = transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.ToTensor()])
    train_data = ImageData(df=df_train, transform=data_transf)
    train_loader = DataLoader(dataset=train_data, batch_size=4)
    submit = pd.read_csv(path + 'sample_submission.csv', converters={'EncodedPixels': lambda e: ' '})
    test_data = ImageData(df=submit, transform=data_transf, subset="test")
    test_loader = DataLoader(dataset=test_data, shuffle=False)
    return train_loader, test_loader


def rle2mask(rle, imgshape):
    width = imgshape[0]
    height = imgshape[1]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return np.flipud(np.rot90(mask.reshape(height, width), k=1))


def make_aug(df_train):
    df_aug = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])
    for index, row in df_train.iterrows():
        img = cv2.imread((path + "train/" + row['ImageId_ClassId']).split('_')[0])
        fname = (path + "train/" + row['ImageId_ClassId']).split('_')[0]
        mask = rle2mask(row['EncodedPixels'], (256, 1600))
        aug = load_aug(img)
        augmented = aug(image=img, mask=mask)
        image_aug = augmented['image']
        mask_aug = augmented['mask']
        fname_aug = fname + '_5'
        cv2.imwrite(fname_aug, image_aug)
        df_aug.loc[index] = [fname_aug, mask2rle(mask_aug)]
    return df_aug


def mask2rle(img):
    tmp = np.rot90(np.flipud(img), k=3)
    rle = []
    lastColor = 0
    startpos = 0
    endpos = 0

    tmp = tmp.reshape(-1,1)
    for i in range( len(tmp) ):
        if (lastColor==0) and tmp[i]>0:
            startpos = i
            lastColor = 1
        elif (lastColor==1)and(tmp[i]==0):
            endpos = i-1
            lastColor = 0
            rle.append( str(startpos)+' '+str(endpos-startpos+1) )
    return " ".join(rle)


def load_aug(img):
    original_height, original_width = img.shape[:2]
    aug = Compose([
        OneOf([RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.5),
               PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)], p=1),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8),
        CLAHE(p=0.8),
        RandomBrightnessContrast(p=0.8),
        RandomGamma(p=0.8)])
    return aug


class ImageData(Dataset):
    def __init__(self, df, transform, subset="train"):
        super().__init__()
        self.df = df
        self.transform = transform
        self.subset = subset

        if self.subset == "train":
            self.data_path = path + 'train/'
        elif self.subset == "test":
            self.data_path = path + 'test/'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fn = self.df['ImageId_ClassId'].iloc[index].split('_')[0]
        img = Image.open(self.data_path + fn)
        img = self.transform(img)

        if self.subset == 'train':
            mask = rle2mask(self.df['EncodedPixels'].iloc[index], (256, 1600))
            mask = transforms.ToPILImage()(mask)
            mask = self.transform(mask)
            return img, mask
        else:
            mask = None
            return img