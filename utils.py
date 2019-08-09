# coding: utf-8
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


path = 'data/'


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