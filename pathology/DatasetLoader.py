import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from glob import glob
from random import choice
import pickle

################
# Dataset Loader
################
class MILFeatureLoader(Dataset):
    def __init__(self, opt, data, aug=False, shuffle_bag=False):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.PatientID = data['ID']
        self.DFS = data['DFS']
        self.OS = data['OS']
        self.DEvent = data['DEvent']
        self.OSEvent = data['OSEvent']
        self.shuffle_bag = shuffle_bag
        self.aug = aug
        self.opt = opt

    def __getitem__(self, index):
        OS = torch.tensor(self.OS[index]).type(torch.FloatTensor)
        DFS = torch.tensor(self.DFS[index]).type(torch.FloatTensor)
        OSEvent = torch.tensor(self.OSEvent[index]).type(torch.FloatTensor)
        DEvent = torch.tensor(self.DEvent[index]).type(torch.FloatTensor)
        if self.aug == False:
            bag_fp = glob(self.opt.pathology_path+self.PatientID[index]+'.val_feature'+'/*.pkl')[0]
        else:
            bag_fp_val = glob(self.opt.pathology_path+self.PatientID[index]+'.val_feature'+'/*.pkl')
            bag_fp_aug = glob(self.opt.pathology_path+self.PatientID[index]+'.train_feature'+'/*.pkl')
            bag_fp = bag_fp_val + bag_fp_aug
            bag_fp = choice(bag_fp)
            
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

        if self.shuffle_bag:
            instance_size = feat.shape[0]
            idx = torch.randperm(instance_size)
            feat = feat[idx]
    
        return {'data': feat, 'OS':OS, 'DFS':DFS, 'OSEvent': OSEvent, 'DEvent': DEvent,
                'PatientID': self.PatientID[index]}

    def __len__(self):
        return len(self.PatientID)
    

def obtain_data(tabel_path, pathology_path):
    pathology_paths = glob(pathology_path+'/*')
    patient_ID=[]
    for pathology_path in pathology_paths:
        start_index = pathology_path.rfind('/') + 1  
        end_index = pathology_path.rfind('.')  
        patient_ID.append(pathology_path[start_index:end_index]) 
    patient_ID = list(set(patient_ID))
    train_tab = pd.read_excel(tabel_path, index_col=0)
    DFS = train_tab.loc[patient_ID, 'DFS'].values
    OS = train_tab.loc[patient_ID, 'OS'].values
    DEvent = train_tab.loc[patient_ID, 'Distant metastasis（no=0；yes=1）'].values
    OSEvent = train_tab.loc[patient_ID, 'Death（no=0；yes=1）'].values
    return {'ID': patient_ID, 'DFS': DFS, 'OS': OS, 'DEvent': DEvent, 'OSEvent': OSEvent}
