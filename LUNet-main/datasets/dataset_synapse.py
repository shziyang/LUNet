import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import torch.nn as nn
import pandas
from transformers import BertTokenizer, BertModel

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def read_text(filename):
    df = pandas.read_excel(filename)
    text = {}
    for i in df.index.values:  # Gets the index of the row number and traverses it
        count = len(df.Description[i].split())
        if count < 9:
            df.Description[i] = df.Description[i] + ' EOF XXX' * (9 - count)
        text[df.Image[i]] = df.Description[i]
    return text  # return dict (key: values)

class ConvTransBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

        self.rowtext = read_text('/home/li/LUNet/data/Synapse/text_english.xlsx')
        self.tokenizer = BertTokenizer.from_pretrained('/home/li/LUNet/BERT')
        self.model = BertModel.from_pretrained('/home/li/LUNet/BERT')
        self.CTBN = ConvTransBN(98, 49)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + "/" + slice_name + '.npz'
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + "/" + slice_name + '.npz'
            data = np.load(data_path, allow_pickle=True)
            image, label = data['image'], data['label']
            image = torch.from_numpy(image.astype(np.float32))
            image = image.permute(2, 0, 1)
            label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)         #图像增强
        #读取文本数据
        text = self.rowtext[int(slice_name)]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=98, return_attention_mask=True, return_token_type_ids=False)
        inputs = {k: v.repeat(2, 1) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = self.CTBN(last_hidden_states)
        sample['text'] = last_hidden_states
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample