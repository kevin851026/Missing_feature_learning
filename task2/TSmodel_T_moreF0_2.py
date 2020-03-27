import torch
import time
import random
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
def get_upload(model):
    testset = Dataset('test.csv',train=False)
    test_loader = DataLoader(testset, batch_size=batch_size)
    model.eval()
    csv_file = open('upload.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Id','Class'])
    for step, (batch) in enumerate(test_loader):
        input_id,input_tensor=[t.to(device) for t in batch]
        logits = model(features = input_tensor)
        topv, topi = logits.topk(1)
        for label,predic in zip(input_id.tolist(),topi.tolist()):
            csv_writer.writerow([label,testset.label_id[predic[0]]])
    csv_file.close()
def get_loss(dataset,model):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    criterion1 = nn.MSELoss()
    train_ls_1 = 0
    for step, (batch) in enumerate(loader):
        input_tensor,miss_tensor,target_tensor=[t.to(device) for t in batch]
        logits = model(features = input_tensor)
        loss1 = criterion1(logits, miss_tensor)
        train_ls_1 += loss1.item()
    return train_ls_1/(step+1)
class Dataset(Dataset):
    def __init__(self, file, train=True, val=True):
        self.train = train
        self.val = val
        self.data = pd.read_csv(file, encoding='utf-8')
        self.label_id = [0,1]
        self.len = len(self.data)
    def __getitem__(self, index):
        if self.train:
            feature = self.data.iloc[index,1:-1]
            feature.drop(['F2','F7','F12'], inplace=True)
            feature = feature.values.astype(np.float32)
            miss_tensor = torch.Tensor([self.data.iloc[index,2],self.data.iloc[index,7],self.data.iloc[index,12]])
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
            feature.drop(['F2','F7','F12'], inplace=True)
            feature = feature.values.astype(np.float32)
            feature = torch.tensor(feature)
            # feature = np.nan_to_num(feature)
            return input_id,feature
    def __len__(self):
        return self.len
class Model(nn.Module):
    def __init__(self, input_size,hidden_size,category_size):
        super(Model, self).__init__()
        self.regression1 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.ReLU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.ReLU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,1),
                    )
        self.regression2 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.ReLU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.ReLU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,1),
                    )
        self.regression3 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.ReLU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.ReLU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,1),
                    )
        self.weight = nn.Parameter(torch.ones(2))
    def forward(self, features):
        regression1 = self.regression1(features)
        regression2 = self.regression2(features)
        regression3 = self.regression3(features)
        return torch.cat((regression1,regression2,regression3),dim=1)

def draw_chart(chart_data,outfile_name):
    plt.figure()
    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    plt.plot(chart_data['epoch'],chart_data['tarin_loss'],label='tarin_loss')
    plt.grid(True,axis="y",ls='--')
    plt.legend(loc= 'best')
    plt.xlabel('epoch',fontsize=20)
    # plt.yticks(np.linspace(0,1,11))
    plt.savefig('./image/'+outfile_name+'_tarin_loss.jpg')
    plt.close('all')
    plt.figure()
    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    plt.plot(chart_data['epoch'],chart_data['val_acc'],label='val_acc')
    plt.plot(chart_data['epoch'],chart_data['train_acc'],label='train_acc')
    plt.grid(True,axis="y",ls='--')
    plt.legend(loc= 'best')
    plt.xlabel('epoch',fontsize=20)
    # plt.yticks(np.linspace(0,1,11))
    plt.savefig('./image/'+outfile_name+'_val_acc.jpg')
    plt.close('all')
    with open('./image/'+outfile_name+'.json','w') as file_object:
        json.dump(chart_data,file_object)
drop_pro = 0.01
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    batch_size = 256
    hidden_size = 128
    drop_pro = 0.01
    learning_rate = 0.001

    torch.manual_seed(3)
    dataset = Dataset('task2_train.csv')
    valset = Dataset('task2_val.csv',val=False)
    train_loader = DataLoader(dataset, batch_size=batch_size)

    model = Model(11, hidden_size, len(dataset.label_id))
    # model.load_state_dict(torch.load('task2_moreF0_2.pkl'))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    min_loss = get_loss(valset,model)
    print(min_loss)
    # get_upload(model)
    # exit()
    criterion1 = nn.MSELoss()
    for epoch in range(15000):
        train_ls_1 = 0
        count = 0
        correct = 0
        model.train()
        for step, (batch) in enumerate(train_loader):
            input_tensor,miss_tensor,target_tensor = [t.to(device) for t in batch]
            logits = model(features = input_tensor)

            loss1 = criterion1(logits, miss_tensor)
            train_ls_1 += loss1.item()

            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

        v_loss = get_loss(valset,model)
        if v_loss < min_loss :
            min_loss = v_loss
            torch.save(model.state_dict(), 'task2_moreF0_2'+'.pkl')
        print('Epoch: ' , str(epoch) , \
              '\ttrain loss: '+str(round(train_ls_1/(step+1),5)),\
              '\ttrain loss: '+str(round(v_loss,5)))