import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import itertools

# Util function


def Label2Arr(cap_list, cuis):
    labels = list(itertools.repeat(0, len(cuis)))
    for i in cap_list:
        for idx, cap in enumerate(cuis):
            if cap == i:
                labels[idx] = 1

    return labels

# Dataset ready for Dataloader


class MultiLabel(Dataset):
    def __init__(self, df, cuis, imgsize, tfm, args, mode='train'):
        self.df = df
        self.tfm = tfm
        self.cuis = cuis
        self.size = imgsize
        self.args = args
        #self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'ID']
        captions = self.df.loc[idx, 'cuis']
        cap_list = captions.split(";")
        labels = Label2Arr(cap_list, self.cuis)
        labels = torch.from_numpy(np.asarray(labels, dtype=float))
        img = cv2.imread(path)
        if self.tfm:
            img = self.tfm(img)

        return img.float(), labels.float()
