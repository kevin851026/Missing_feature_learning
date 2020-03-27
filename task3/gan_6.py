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
        input_tensor,z,all_tensor,target_tensor = [t.to(device) for t in batch]
        G_sample = model.G(input_tensor,z)
        logits = model.C(G_sample)
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
        input_tensor,z,all_tensor,target_tensor = [t.to(device) for t in batch]
        G_sample = model.G(input_tensor,z)
        logits = model.C(G_sample)
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
        self.z = np.random.uniform(0,1,size=[self.len])
    def __getitem__(self, index):
        if self.train:
            feature = self.data.iloc[index,1:-1]
            feature.loc['F1'] = 0
            feature = feature.values.astype(np.float32)
            all_feature = self.data.iloc[index,1:-1]
            category = self.data.iloc[index,-1]
            try:
                label_tensor = self.label_id.index(category)
            except:
                print(self.data.iloc[index].values)
            feature_tensor = torch.tensor(feature)
            all_tensor = torch.tensor(all_feature)
            z = self.z[index]
            return feature_tensor,z,all_tensor,label_tensor
        else:
            input_id = self.data.iloc[index,0]
            feature = self.data.iloc[index,1:]
            feature.loc['F1'] = 0
            feature = feature.values.astype(np.float32)
            feature = torch.tensor(feature)
            return input_id,feature
    def __len__(self):
        return self.len
class Generator(nn.Module):
    def __init__(self,input_size):
        super(Generator, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 256)
        self.fc5 = torch.nn.Linear(256, 128)
        self.fc6 = torch.nn.Linear(128, input_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.weight = nn.Parameter(torch.ones(2))
        self.init_weight()

    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

    def forward(self, features, z):
        # features = features.clone()
        # features.T[0] += z
        out = self.relu(self.fc1(features))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.relu(self.fc5(out))
        out = self.fc6(out) # [0,1] Probability Output
#         out = self.fc3(out)

        return out
class Discriminator(nn.Module):
    def __init__(self,input_size):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()
    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, features):
        out = self.relu(self.fc1(features))
        out = self.relu(self.fc2(out))
#         out = self.sigmoid(self.fc3(out)) # [0,1] Probability Output
        out = self.fc3(out)
        return out
class Classifier(nn.Module):
    def __init__(self, input_size,hidden_size,category_size):
        super(Classifier, self).__init__()
        self.fc1 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.ReLU(),
                        nn.Dropout(drop_pro)
                    )
        self.fc2 =  nn.Sequential(
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.ReLU(),
                        nn.Dropout(drop_pro)
                    )
        self.classifier =  nn.Sequential(
                        nn.Linear(hidden_size*2,category_size),
                    )
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, features):
        logits = self.fc1(features)
        logits = self.fc2(logits)
        logits = self.classifier(logits)
        # logits = self.softmax(logits)
        return logits
class Model(nn.Module):
    def __init__(self, input_size,hidden_size,category_size):
        super(Model, self).__init__()
        self.D = Discriminator(input_size)
        self.G = Generator(input_size)
        self.C = Classifier(input_size, hidden_size, category_size)
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
    hidden_size = 128
    drop_pro = 0.1
    learning_rate = 0.001
    alpha = 1
    torch.manual_seed(3)
    dataset = Dataset('task1_re_train.csv')
    valset = Dataset('task1_re_val.csv',val=False)
    train_loader = DataLoader(dataset, batch_size=batch_size)

    model = Model(9, hidden_size, len(dataset.label_id))
    model = model.to(device)
    # model.G.load_state_dict(torch.load('task3_gan_6_G.pkl'))
    # model.D.load_state_dict(torch.load('task3_gan_6_D.pkl'))
    # model.C.load_state_dict(torch.load('task3_gan_6_C.pkl'))
    optimizer_d = optim.Adam(model.D.parameters(), lr=learning_rate,weight_decay=1e-5)
    optimizer_g = optim.Adam(model.G.parameters(), lr=learning_rate,weight_decay=1e-5)
    optimizer_c = optim.Adam(model.C.parameters(), lr=learning_rate,weight_decay=1e-5)
    max_acc = get_acc(valset,model)
    print(max_acc)
    # get_upload(model,classifier)
    # exit()
    chart_data={"tarin_loss":[],"val_acc":[],"train_acc":[],"epoch":[]}
    bce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    for epoch in range(15000):
        discriminator_loss = 0
        generator_loss = 0
        g_mse_loss = 0
        classifier_loss =0
        count = 0
        correct = 0
        model.train()
        for step, (batch) in enumerate(train_loader):
            input_tensor,z,all_tensor,target_tensor = [t.to(device) for t in batch]

            G_sample = model.G(input_tensor,z)
            D_prob = model.D(G_sample.detach())
            D_loss = bce_loss(D_prob,torch.ones(input_tensor.shape[0],dtype=torch.long,device=device))
            # print(G_sample)
            # print(all_tensor)
            # exit()
            D_prob = model.D(all_tensor)
            D_loss += bce_loss(D_prob,torch.zeros(input_tensor.shape[0],dtype=torch.long,device=device))
            # if D_loss.item() >1 or epoch<10 or epoch%10==0:
            optimizer_d.zero_grad()
            D_loss.backward()
            optimizer_d.step()
            discriminator_loss += D_loss.item()

            G_sample = model.G(input_tensor,z)
            D_prob = model.D(G_sample)
            G_loss1 = bce_loss(D_prob,torch.zeros(input_tensor.shape[0],dtype=torch.long,device=device))
            G_mse_loss = mse_loss(G_sample.transpose(0,1)[:1].transpose(0,1),all_tensor.transpose(0,1)[:1].transpose(0,1))
            # G_loss = 0.5*G_loss1/model.G.weight[0]**2 + 0.5*G_mse_loss/model.G.weight[1]**2 + torch.log(model.G.weight.prod())
            G_loss = G_loss1 + alpha*G_mse_loss
            # print(g_frequence)
            optimizer_g.zero_grad()
            G_loss.backward()
            optimizer_g.step()
            generator_loss += G_loss1.item()
            g_mse_loss += G_mse_loss.item()

            G_sample = model.G(input_tensor,z)
            logits = model.C(G_sample)
            topv, topi = logits.topk(1)
            for predic, label in zip(topi.tolist(),target_tensor.tolist()):
                if predic[0] == label:
                    correct+=1
                count+=1
            c_loss = bce_loss(logits, target_tensor)
            classifier_loss += c_loss.item()
            optimizer_c.zero_grad()
            optimizer_g.zero_grad()
            c_loss.backward()
            optimizer_g.step()
            optimizer_c.step()
        # t_acc = get_acc(dataset,model)
        v_acc = get_acc(valset,model)
        # if v_acc > max_acc :
        #     max_acc = v_acc
        torch.save(model.C.state_dict(),'task3_gan_6_C.pkl')
        torch.save(model.G.state_dict(), 'task3_gan_6_G.pkl')
        torch.save(model.D.state_dict(), 'task3_gan_6_D.pkl')
        weight_sum = model.G.weight[0].item()+model.G.weight[1].item()
        print('Epoch: {}'.format(epoch),'  '\
            ,'D_loss: {:.4}'.format(discriminator_loss/step)\
            ,'G_loss: {:.4}'.format(generator_loss/step)\
            ,'mse_loss: {:.4}'.format(g_mse_loss/step),'\t'\
            ,'{:.4}'.format(model.G.weight[0].item()/weight_sum)
            ,'{:.4}'.format(model.G.weight[1].item()/weight_sum),'\t'
            ,'train loss: {:.4}'.format(classifier_loss/(step+1))
            ,'train acc: {:.4}'.format(correct/count)
            ,'val acc: {:.4}'.format(v_acc)
            )
        chart_data['epoch'].append(epoch)
        chart_data['tarin_loss'].append(classifier_loss/(step+1))
        chart_data['train_acc'].append(correct/count)
        chart_data['val_acc'].append(v_acc)
        draw_chart(chart_data,'task3_gan_6')