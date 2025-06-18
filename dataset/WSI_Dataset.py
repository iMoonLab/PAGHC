from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
import pandas as pd
import random
from torch_geometric.data import Data
from sklearn.metrics.pairwise import euclidean_distances
from model.utils.WSI_Graph_Construction import pt2graph
import torch_geometric.transforms as T
from torch_geometric.transforms import Polar

class SlidePatch(Dataset):
    def __init__(self, data_dict: dict, survival_time_max, survival_time_min, CT_ft_file=None):
        super().__init__()
        self.st_max = float(survival_time_max)
        self.st_min = float(survival_time_min)
        self.count = 0
        self.id_list = list(data_dict.keys())
        self.data_dict = data_dict
        # for key in self.id_list:
        #         self.data_dict[key]['survival_time']
        self.CT_ft_file = CT_ft_file
        if CT_ft_file is not None:
            with open(CT_ft_file,'rb') as f:
                CT_feature = pickle.load(f)
            for key in self.id_list:
                self.data_dict[key]['CT_ft'] = CT_feature[self.data_dict[key]['radiology']]


    def __getitem__(self, idx: int):
        id = self.id_list[idx]
        self.count += 1
        fts = torch.tensor(np.load(self.data_dict[id]['ft_dir'])).float() #self.data_dict[id]['fts']
        ti = torch.tensor(self.data_dict[id]['survival_time']).float()
        survival_time = ti/365 #self.st_max #(self.st_min * (self.st_max - ti))/ (ti * (self.st_max - self.st_min)) #ti/self.st_max #  /self.st_max #
        status = torch.tensor(self.data_dict[id]['status'])

        with open(self.data_dict[id]['patch_coors'], 'rb') as f:
            coors = pickle.load(f)
            coors = torch.Tensor(coors)
        
        return fts, survival_time, status, coors, id #, stage, stage_t, stage_m, stage_n

    def __len__(self) -> int:
        # print(len(self.id_list))
        return len(self.id_list)



class SlidePatchCT(Dataset):
    def __init__(self, data_dict: dict, survival_time_max, survival_time_min, CT_ft_file=None):
        super().__init__()
        self.st_max = float(survival_time_max)
        self.st_min = float(survival_time_min)
        self.count = 0
        self.id_list = list(data_dict.keys())
        self.data_dict = data_dict
        # for key in self.id_list:
        #         self.data_dict[key]['survival_time']
        self.CT_ft_file = CT_ft_file
        if CT_ft_file is not None:
            with open(CT_ft_file, 'rb') as f:
                CT_feature = pickle.load(f)
            for key in self.id_list:
                self.data_dict[key]['CT_ft'] = CT_feature[self.data_dict[key]['radiology']]

    def __getitem__(self, idx: int):
        id = self.id_list[idx]
        self.count += 1
        fts = torch.tensor(np.load(self.data_dict[id]['ft_dir'])).float()  # self.data_dict[id]['fts']
        ti = torch.tensor(self.data_dict[id]['survival_time']).float()
        survival_time = ti / 365  # self.st_max #(self.st_min * (self.st_max - ti))/ (ti * (self.st_max - self.st_min)) #ti/self.st_max #  /self.st_max #
        status = torch.tensor(self.data_dict[id]['status'])
        with open(self.data_dict[id]['patch_coors'], 'rb') as f:
            coors = pickle.load(f)
            coors = torch.Tensor(coors)

        if 'axial' in self.data_dict[id].keys():
            axial = torch.tensor(self.data_dict[id]['axial']).float()
            sagittal = torch.tensor(self.data_dict[id]['sagittal']).float()
            coronal = torch.tensor(self.data_dict[id]['coronal']).float()
            return fts, survival_time, status, sagittal, coronal, coors, id

        return fts, survival_time, status, coors, id

    def __len__(self) -> int:
        # print(len(self.id_list))
        return len(self.id_list)
