import torch
import time
import json
import random
import csv
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
        logits,regression = model(features = input_tensor)
        topv, topi = logits.topk(1)
        for label,predic in zip(input_id.tolist(),topi.tolist()):
            csv_writer.writerow([label,testset.label_id[predic[0]]])
    csv_file.close()
def get_acc(dataset,model):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    count = 0
    correct = 0

    for step, (batch) in enumerate(loader):
        input_tensor,miss_tensor,target_tensor=[t.to(device) for t in batch]
        logits,regression = model(features = input_tensor)
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
            feature.drop(['F1'], inplace=True)
            feature = feature.values.astype(np.float32)
            miss_feature = self.data.iloc[index,1:-1]
            category = self.data.iloc[index,-1]
            try:
                label_tensor = self.label_id.index(category)
            except:
                print(self.data.iloc[index].values)
            feature_tensor = torch.tensor(feature)
            miss_tensor = torch.tensor(miss_feature)
            mask = torch.Tensor([1]*len(feature))
            # if self.val:
            #     for i in range(random.randint(0,3)):
            #         mask[random.randint(0,len(mask)-1)] = 0
            feature_tensor = feature_tensor*mask
            return feature_tensor,miss_tensor,label_tensor
        else:
            input_id = self.data.iloc[index,0]
            feature = self.data.iloc[index,1:]
            feature.drop(['F1'], inplace=True)
            feature = feature.values.astype(np.float32)
            feature = torch.tensor(feature)
            return input_id,feature
    def __len__(self):
        return self.len
class Model(nn.Module):
    def __init__(self, input_size,hidden_size,category_size):
        super(Model, self).__init__()
        self.fc1 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.fc2 =  nn.Sequential(
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.fc3 =  nn.Sequential(
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.fc4 =  nn.Sequential(
                        nn.Linear(input_size*2+1,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.fc5 =  nn.Sequential(
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.regression =  nn.Sequential(
                        nn.Linear(hidden_size*2,input_size+1),
                    )
        self.classifier =  nn.Sequential(
                        nn.Linear(hidden_size*2,category_size),
                    )
        self.softmax = nn.LogSoftmax(dim=1)
        self.weight = nn.Parameter(torch.ones(2))
    def forward(self, features):
        logits = self.fc1(features)
        logits = self.fc2(logits)
        logits = self.fc3(logits)
        regression = self.regression(logits)
        logits = self.fc4(torch.cat((features,regression),dim=1))
        logits = self.fc5(logits)
        logits = self.classifier(logits)
        # logits = self.softmax(logits)
        return logits,regression
     # 1              5
     # 2              2
     # 3              3
     # 4              4
#  regression     classifier
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
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    batch_size = 256
    hidden_size = 64
    drop_pro = 0.1
    learning_rate = 0.001

    torch.manual_seed(1010)
    trainset = Dataset('task1_re_train.csv')
    valset = Dataset('task1_re_val.csv',val=False)
    train_loader = DataLoader(trainset, batch_size=batch_size)

    model = Model(9-1, hidden_size, len(trainset.label_id))
    model.load_state_dict(torch.load('task3_combine23_ae_1.pkl'))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    max_acc = get_acc(valset,model)
    print(max_acc)
    get_upload(model)
    exit()
    chart_data = {"tarin_loss":[],"val_acc":[],"train_acc":[],"epoch":[]}
    for epoch in range(15000):
        train_ls_1 = 0
        train_ls_2 = 0
        total_ls = 0
        model.train()
        # print(time.asctime( time.localtime(time.time()) ))
        count = 0
        correct = 0
        for step, (batch) in enumerate(train_loader):
            input_tensor,miss_tensor,target_tensor = [t.to(device) for t in batch]
            logits,regression = model(features = input_tensor)
            topv, topi = logits.topk(1)
            for predic, label in zip(topi.tolist(),target_tensor.tolist()):
                if predic[0] == label:
                    correct+=1
                count+=1
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.MSELoss()
            loss1 = criterion1(logits, target_tensor)
            loss2 = criterion2(regression, miss_tensor)
            train_ls_1 += loss1.item()
            train_ls_2 += loss2.item()
            optimizer.zero_grad()
            loss = 0.5*loss1/model.weight[0]**2 + 0.5*loss2/model.weight[1]**2 + torch.log(model.weight.prod())
            total_ls = loss.item()
            loss.backward()
            optimizer.step()
        v_acc = get_acc(valset,model)
        if v_acc > max_acc :
            max_acc = v_acc
            torch.save(model.state_dict(), 'task3_combine23_ae_1'+'.pkl')
        weight_sum = model.weight[0].item()+model.weight[1].item()
        print('Epoch: ' , str(epoch) , \
              '\ttrain loss: ' , round(train_ls_1/(step+1),5), ' , ', round(train_ls_2/(step+1),5),\
              '\ttrain acc: '+str(round(correct/count,5))+' val acc: '+str(round(v_acc,5)),\
              ' ',round(model.weight[0].item()/weight_sum,4) , ' ' , round(model.weight[1].item()/weight_sum,4))
        chart_data['epoch'].append(epoch)
        chart_data['tarin_loss'].append(total_ls/(step+1))
        chart_data['train_acc'].append(correct/count)
        chart_data['val_acc'].append(v_acc)
        draw_chart(chart_data,'task3_combine23_ae_1')