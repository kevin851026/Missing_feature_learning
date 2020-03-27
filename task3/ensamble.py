import torch
import time
import random
import csv
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from TSmodel_T_moreF9 import Model as moreF9
from TSmodel_T_moreF7 import Model as moreF7
from TSmodel_T_moreF7_2 import Model as moreF7_2
from TSmodel_T_moreF7_3 import Model as moreF7_3
from TSmodel_T_moreF5 import Model as moreF5
from TSmodel_T_moreF3 import Model as moreF3
from TSmodel_T_moreF0 import Model as moreF0
from TSmodel_T_combine23 import Model as combine23
def get_acc(dataset,model):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    count = 0
    correct = 0
    for step, (batch) in enumerate(loader):
        input_tensor,miss_tensor,target_tensor=[t.to(device) for t in batch]
        logits = model(features = input_tensor)
        topv, topi = logits.topk(1)
        for predic, label in zip(topi.tolist(),target_tensor.tolist()):
            if predic[0] == label:
                correct+=1
            count+=1
    return correct/count
class Dataset(Dataset):
    def __init__(self, file, train=True, val=True):
        self.train = train
        self.val = val
        self.data = pd.read_csv(file, encoding='utf-8')
        self.label_id = ['A','B','C','D','E','F','G','H', 'I', 'J', 'K', 'L']
        self.len = len(self.data)
    def __getitem__(self, index):
        if self.train:
            feature = self.data.iloc[index,1:-1]
            # feature.drop(['F1'], inplace=True)
            feature = feature.values.astype(np.float32)
            miss_tensor = self.data.iloc[index,1]
            category = self.data.iloc[index,-1]
            try:
                label_tensor = self.label_id.index(category)
            except:
                print(self.data.iloc[index].values)
            feature_tensor = torch.tensor(feature)
            return feature_tensor,miss_tensor,label_tensor
        else:
            input_id = self.data.iloc[index,0]
            feature = self.data.iloc[index,1:]
            # feature.drop(['F1'], inplace=True)
            feature = feature.values.astype(np.float32)
            feature = torch.tensor(feature)
            return input_id,feature
    def __len__(self):
        return self.len
class Model(nn.Module):
    def __init__(self, input_size,hidden_size,category_size):
        super(Model, self).__init__()
        self.moreF9 = moreF9(9,128,12)
        self.moreF7 = moreF7(9,128,12)
        self.moreF7_2 = moreF7_2(9,128,12)
        self.moreF7_3 = moreF7_3(9,128,12)
        self.moreF5 = moreF5(9,128,12)
        self.moreF3 = moreF3(9,128,12)
        self.moreF0 = moreF0(9,128,12)
        self.combine23 = combine23(9,128,12)

        self.moreF9.load_state_dict(torch.load('task3_moreF9.pkl'))
        self.moreF7.load_state_dict(torch.load('task3_moreF7.pkl'))
        self.moreF7_2.load_state_dict(torch.load('task3_moreF7_2.pkl'))
        self.moreF7_3.load_state_dict(torch.load('task3_moreF7_3.pkl'))
        self.moreF5.load_state_dict(torch.load('task3_moreF5.pkl'))
        self.moreF3.load_state_dict(torch.load('task3_moreF3.pkl'))
        self.moreF0.load_state_dict(torch.load('task3_moreF0.pkl'))
        self.combine23.load_state_dict(torch.load('task3_combine23.pkl'))
    def forward(self, features):
        F9 = self.moreF9(features)
        F7 = self.moreF7(features)
        F7_2 = self.moreF7_2(features)
        F7_3 = self.moreF7_3(features)
        F5 = self.moreF5(features)
        F3 = self.moreF3(features)
        F0 = self.moreF0(features)
        return F9 + F7_2 + F5 + F3 + F0
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    batch_size = 256
    hidden_size = 128
    drop_pro = 0.01
    learning_rate = 0.001

    torch.manual_seed(3)
    dataset = Dataset('task1_re_train.csv')
    valset = Dataset('task1_re_val.csv',val=False)
    train_loader = DataLoader(dataset, batch_size=batch_size)

    model = Model(9, hidden_size, len(dataset.label_id))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    max_acc = get_acc(valset,model)
    print(max_acc)
    exit()