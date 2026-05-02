### data_loaders.py
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
from glob import glob
from random import choice
import pickle
################
# Dataset Loader
################
class MILDatasetLoader(Dataset):
    def __init__(self, opt, data):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.PatientID = data['ID']
        self.DFS = data['DFS']
        self.OS = data['OS']
        
        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(opt.FINE_SIZE),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        OS = torch.tensor(self.OS[index]).type(torch.FloatTensor)
        DFS = torch.tensor(self.DFS[index]).type(torch.FloatTensor)
        bag_fp = glob("./data/*/*/patch/1024/"+self.PatientID[index]+'.ndpi'+'/*.png')

        for bag in range(len(bag_fp)):
            if bag == 0: 
                X = torch.unsqueeze(self.transforms(Image.open(bag_fp[bag]).convert('RGB')), 0)
            else: 
                X = torch.cat((X, torch.unsqueeze(self.transforms(Image.open(bag_fp[bag]).convert('RGB')), 0)), 0)
        return {'data': X, 'OS':OS, 'DFS':DFS,
                'PatientID': self.PatientID[index]}

    def __len__(self):
        return len(self.PatientID)


class MILFeatureLoader(Dataset):
    def __init__(self, opt, data, path, aug=False, shuffle_bag=False):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.PatientID = data['ID']
        self.path = path
        self.DFS = data['DFS']
        self.OS = data['OS']
        self.DEvent = data['DEvent']
        self.OSEvent = data['OSEvent']
        self.shuffle_bag = shuffle_bag
        self.aug = aug

    def __getitem__(self, index):
        OS = torch.tensor(self.OS[index]).type(torch.FloatTensor)
        DFS = torch.tensor(self.DFS[index]).type(torch.FloatTensor)
        OSEvent = torch.tensor(self.OSEvent[index]).type(torch.FloatTensor)
        DEvent = torch.tensor(self.DEvent[index]).type(torch.FloatTensor)
        if self.aug == False:
            #print(self.PatientID,index)
            #print("/data3/louwei/MedComm/data/efficientnet/" + self.path + '/' + self.PatientID[index]+'.val_feature'+'/*.pkl')
            #print(glob("/data3/louwei/MedComm/data/efficientnet/" + self.path + '/' + self.PatientID[index]+'.val_feature'+'/*.pkl')[0])
            bag_fp = glob("/data3/louwei/MedComm/data/efficientnet/" + self.path + '/' + self.PatientID[index]+'.val_feature'+'/*.pkl')[0]
        else:
            
            bag_fp_val = glob("/data3/louwei/MedComm/data/efficientnet/" + self.path + '/' +self.PatientID[index]+'.val_feature'+'/*.pkl')
            bag_fp_aug = glob("/data3/louwei/MedComm/data/efficientnet/" + self.path + '/' + self.PatientID[index]+'.train_feature'+'/*.pkl')
            #print("/data3/louwei/MedComm/data/" + self.path +self.PatientID[index]+'.train_feature')
            bag_fp = bag_fp_val + bag_fp_aug
            bag_fp = choice(bag_fp)
        #print('1111',bag_fp)
        with open(bag_fp, 'rb') as f: 
            bag_feat_list_obj = pickle.load(f)
            bag_feat = []
            for feat_dict in bag_feat_list_obj:
                aug_feat = feat_dict['feature']
                bag_feat.append(aug_feat)


        feat = np.vstack(bag_feat)
        if isinstance(feat, np.ndarray):
            feat = feat.astype(np.float)
        elif isinstance(feat, torch.Tensor):
            feat = feat.float()
        else:
            raise ValueError('Data type not understood')
       #print(feat.shape,self.PatientID[index])
        #if self.PatientID[index] == 'SICK0012503':
            #print(feat.shape,self.PatientID[index])
        
        if self.shuffle_bag:
            instance_size = feat.shape[0]
            idx = torch.randperm(instance_size)
            feat = feat[idx]
    
        return {'data': feat, 'OS':OS, 'DFS':DFS, 'OSEvent': OSEvent, 'DEvent': DEvent,
                'PatientID': self.PatientID[index]}

    def __len__(self):
        return len(self.PatientID)