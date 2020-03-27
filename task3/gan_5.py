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
def get_upload(model):
    testset = Dataset('test.csv',train=False)
    test_loader = DataLoader(testset, batch_size=batch_size)
    model.eval()
    csv_file = open('upload.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Id','Class'])
    for step, (batch) in enumerate(test_loader):
        input_id,input_tensor=[t.to(device) for t in batch]
        new_feature = model.G(features = input_tensor)
        logits = model.C(features = torch.cat((input_tensor,new_feature),dim=1))
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
        new_feature = model.G(features = input_tensor)
        logits = model.C(features = torch.cat((input_tensor,new_feature),dim=1))
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
            feature.drop(['F1','F5','F8'], inplace=True)
            feature = feature.values.astype(np.float32)
            miss_feature = self.data.loc[index,['F1','F5','F8']]
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
            feature.drop(['F1','F5','F8'], inplace=True)
            feature = feature.values.astype(np.float32)
            feature = torch.tensor(feature)
            return input_id,feature
    def __len__(self):
        return self.len
class Generator(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(Generator, self).__init__()
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
        self.regression =  nn.Sequential(
                        nn.Linear(hidden_size*2,3),
                    )
    def forward(self, features):
        regression = self.fc1(features)
        regression = self.fc2(regression)
        regression = self.fc3(regression)
        regression = self.regression(regression)
        return regression
class Discriminator(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 =  nn.Sequential(
                        nn.Linear(3,hidden_size*2),
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
        self.classifier =  nn.Sequential(
                        nn.Linear(hidden_size*2,2),
                    )
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, features):
        logits = self.fc1(features)
        logits = self.fc2(logits)
        logits = self.fc3(logits)
        logits = self.classifier(logits)
        logits = self.softmax(logits)
        return logits
class Classifier(nn.Module):
    def __init__(self, input_size,hidden_size,category_size):
        super(Classifier, self).__init__()
        self.fc1 =  nn.Sequential(
                        nn.Linear(input_size+3,hidden_size*2),
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
        self.classifier =  nn.Sequential(
                        nn.Linear(hidden_size*2,category_size),
                    )
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, features):
        logits = self.fc1(features)
        logits = self.fc2(logits)
        logits = self.fc3(logits)
        logits = self.classifier(logits)
        logits = self.softmax(logits)
        return logits
class Model(nn.Module):
    def __init__(self, input_size,hidden_size,category_size):
        super(Model, self).__init__()
        self.D = Discriminator(input_size,hidden_size)
        self.G = Generator(input_size,hidden_size)
        self.C = Classifier(input_size,hidden_size,category_size)
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

    batch_size = 1024
    hidden_size = 128
    drop_pro = 0.1
    learning_rate = 0.001

    torch.manual_seed(3)
    dataset = Dataset('task1_re_train.csv')
    valset = Dataset('task1_re_val.csv',val=False)
    train_loader = DataLoader(dataset, batch_size=batch_size)

    model = Model(9-3, hidden_size, len(dataset.label_id))
    # model.load_state_dict(torch.load('task3_gan_4.pkl'))
    # model.D.load_state_dict(torch.load('task3_ganD_2.pkl'))
    # model.G.load_state_dict(torch.load('task3_ganG_2.pkl'))
    model = model.to(device)
    optimizer_d = optim.Adam(model.D.parameters(), lr=learning_rate,weight_decay=1e-5)
    optimizer_g = optim.Adam(model.G.parameters(), lr=learning_rate,weight_decay=1e-5)
    optimizer_c = optim.Adam(model.C.parameters(), lr=learning_rate,weight_decay=1e-5)
    max_acc = get_acc(valset,model)
    print(max_acc)
    # get_upload(model)
    # exit()
    chart_data={"tarin_loss":[],"val_acc":[],"train_acc":[],"epoch":[]}
    criterion1 = nn.CrossEntropyLoss()
    for epoch in range(15000):
        discriminator_loss = 0
        generator_loss = 0
        classifier_loss =0
        count = 0
        correct = 0
        model.train()
        for d_frequence in range(1):
            #---- Train D on real data-------
            for step, (batch) in enumerate(train_loader):
                input_tensor,miss_tensor,target_tensor = [t.to(device) for t in batch]
                d_real_logit = model.D(features = miss_tensor)
                d_real_loss = criterion1(d_real_logit,torch.ones(input_tensor.shape[0],dtype=torch.long,device=device))
                discriminator_loss += d_real_loss.item()
                optimizer_d.zero_grad()
                d_real_loss.backward()
                optimizer_d.step()
            #---- Train D on G's data-------
            for step, (batch) in enumerate(train_loader):
                input_tensor,miss_tensor,target_tensor = [t.to(device) for t in batch]
                d_fake_data = model.G(features = input_tensor)
                d_fake_logit = model.D(features = d_fake_data)
                d_fake_loss = criterion1(d_fake_logit,torch.zeros(input_tensor.shape[0],dtype=torch.long,device=device))
                discriminator_loss += d_fake_loss.item()
                optimizer_d.zero_grad()
                d_fake_loss.backward()
                optimizer_d.step()
        torch.save(model.D.state_dict(), 'task3_ganD_5'+'.pkl')
        #---- Train G to fool D-------
        for g_frequence in range(1):
            for step, (batch) in enumerate(train_loader):
                input_tensor,miss_tensor,target_tensor = [t.to(device) for t in batch]
                g_fake_data = model.G(features = input_tensor)
                dg_fake_logit = model.D(features = g_fake_data)
                g_loss = criterion1(dg_fake_logit,torch.ones(input_tensor.shape[0],dtype=torch.long,device=device))
                generator_loss += g_loss.item()
                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()
        torch.save(model.G.state_dict(), 'task3_ganG_5'+'.pkl')

    #     print('Epoch: '+str(epoch)+\
    #           '\tD_loss: '+str(round(discriminator_loss/(step+1),5))+\
    #           '\tG_loss: '+str(round(generator_loss/(step+1),5)))
            #---- Train C with G's output-------
        for step, (batch) in enumerate(train_loader):
            input_tensor,miss_tensor,target_tensor = [t.to(device) for t in batch]
            new_feature = model.G(features = input_tensor)
            logits = model.C(features = torch.cat((input_tensor,new_feature),dim=1))
            topv, topi = logits.topk(1)
            for predic, label in zip(topi.tolist(),target_tensor.tolist()):
                if predic[0] == label:
                    correct+=1
                count+=1
            c_loss = criterion1(logits, target_tensor)
            classifier_loss += c_loss.item()
            optimizer_c.zero_grad()
            optimizer_g.zero_grad()
            c_loss.backward()
            optimizer_c.step()
            optimizer_g.step()
        # t_acc = get_acc(dataset,model)
        v_acc = get_acc(valset,model)
        if v_acc > max_acc :
            max_acc = v_acc
            torch.save(model.state_dict(), 'task3_gan_5'+'.pkl')
        print('Epoch: '+str(epoch)+\
              '\ttrain loss: '+str(round(classifier_loss/(step+1),5))+\
              '\ttrain acc: '+str(round(correct/count,5))+' val acc: '+str(round(v_acc,5)))
        chart_data['epoch'].append(epoch)
        chart_data['tarin_loss'].append(classifier_loss/(step+1))
        chart_data['train_acc'].append(correct/count)
        chart_data['val_acc'].append(v_acc)
        draw_chart(chart_data,'task3_gan_5')