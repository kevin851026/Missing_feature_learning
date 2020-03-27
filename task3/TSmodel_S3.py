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
def get_upload(model):
    testset = Dataset('test.csv',train=False)
    test_loader = DataLoader(testset, batch_size=batch_size)
    model.eval()
    csv_file = open('upload.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Id','Class'])
    for step, (batch) in enumerate(test_loader):
        input_id,input_tensor=[t.to(device) for t in batch]
        logits,generate = model(features = input_tensor)
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
        logits,generate = model(features = input_tensor)
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
            all_feautre = self.data.iloc[index,1:-1]
            all_feautre = all_feautre.values.astype(np.float32)
            category = self.data.iloc[index,-1]
            try:
                label_tensor = self.label_id.index(category)
            except:
                print(self.data.iloc[index].values)
            feature_tensor = torch.tensor(feature)
            all_tensor = torch.tensor(all_feautre)

            return feature_tensor,all_tensor,label_tensor
        else:
            input_id = self.data.iloc[index,0]
            feature = self.data.iloc[index,1:]
            feature.drop(['F1'], inplace=True)
            feature = feature.values.astype(np.float32)
            feature = torch.tensor(feature)
            return input_id,feature
    def __len__(self):
        return self.len
class Teacher(nn.Module):
    def __init__(self, input_size,hidden_size,category_size):
        super(Teacher, self).__init__()
        self.fc1 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.regression1 =  nn.Sequential(
                        nn.Linear(hidden_size*2,1),
                    )
        self.fc2 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.regression2 =  nn.Sequential(
                        nn.Linear(hidden_size*2,1),
                    )
        self.fc3 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.regression3 =  nn.Sequential(
                        nn.Linear(hidden_size*2,1),
                    )
        self.fc4 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.regression4 =  nn.Sequential(
                        nn.Linear(hidden_size*2,1),
                    )
        self.fc5 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.regression5 =  nn.Sequential(
                        nn.Linear(hidden_size*2,1),
                    )
        self.fc6 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.regression6 =  nn.Sequential(
                        nn.Linear(hidden_size*2,1),
                    )
        self.fc7 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.regression7 =  nn.Sequential(
                        nn.Linear(hidden_size*2,1),
                    )
        self.fc8 =  nn.Sequential(
                        nn.Linear(input_size+7,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.fc9 =  nn.Sequential(
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro)
                    )
        self.classifier =  nn.Sequential(
                        nn.Linear(hidden_size*2,category_size),
                    )
        self.weight = nn.Parameter(torch.ones(2))
    def forward(self, features):
        regression1 = self.regression1(self.fc1(features))
        regression2 = self.regression2(self.fc2(features))
        regression3 = self.regression3(self.fc3(features))
        regression4 = self.regression4(self.fc4(features))
        regression5 = self.regression5(self.fc5(features))
        regression6 = self.regression6(self.fc6(features))
        regression7 = self.regression7(self.fc7(features))
        new_features = features.clone()
        new_features = torch.cat((new_features, regression1,regression2,regression3,regression4,regression5,regression6,regression7),dim=1)
        logits = self.fc8(new_features)
        logits = self.fc9(logits)
        logits = self.classifier(logits)
        return logits,torch.cat((regression1,regression2,regression3,regression4,regression5,regression6,regression7),dim=1)
class Model(nn.Module):
    def __init__(self, input_size,hidden_size,category_size):
        super(Model, self).__init__()
        self.generate1 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,1),
                    )
        self.generate2 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,1),
                    )
        self.generate3 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,1),
                    )
        self.generate4 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,1),
                    )
        self.generate5 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,1),
                    )
        self.generate6 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,1),
                    )
        self.generate7 =  nn.Sequential(
                        nn.Linear(input_size,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,1),
                    )
        self.classifier =  nn.Sequential(
                        nn.Linear(input_size+7,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,hidden_size*2),
                        nn.SELU(),
                        nn.Dropout(drop_pro),
                        nn.Linear(hidden_size*2,category_size),
                    )
        self.weight = nn.Parameter(torch.ones(2))
    def forward(self, features):
        generate1 = self.generate1(features)
        generate2 = self.generate2(features)
        generate3 = self.generate3(features)
        generate4 = self.generate4(features)
        generate5 = self.generate5(features)
        generate6 = self.generate6(features)
        generate7 = self.generate7(features)
        expand_features = features.clone()
        expand_features = torch.cat((expand_features, generate1,generate2,generate3,generate4,generate5,generate6,generate7),dim=1)
        logits = self.classifier(expand_features)
        return logits,torch.cat((generate1,generate2,generate3,generate4,generate5,generate6,generate7),dim=1)

class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num

        return loss
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

    batch_size = 512
    hidden_size = 128
    drop_pro = 0.1
    learning_rate = 0.001

    torch.manual_seed(3)
    dataset = Dataset('task1_re_train.csv')
    valset = Dataset('task1_re_val.csv',val=False)
    train_loader = DataLoader(dataset, batch_size=batch_size)

    teacher = Teacher(9, hidden_size, len(dataset.label_id))
    model = Model(9-1, hidden_size, len(dataset.label_id))

    teacher.load_state_dict(torch.load('task3_moreF7_2.pkl'))
    model.load_state_dict(torch.load('task3_student_3.pkl'))

    teacher = teacher.to(device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    max_acc = get_acc(valset,model)
    print(max_acc)
    get_upload(model)
    exit()
    chart_data={"tarin_loss":[],"val_acc":[],"train_acc":[],"epoch":[]}
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    criterion3 = softCrossEntropy()
    for epoch in range(15000):
        train_ls_1 = 0
        count = 0
        correct = 0
        teacher.eval()
        model.train()
        for step, (batch) in enumerate(train_loader):
            input_tensor,all_tensor,target_tensor = [t.to(device) for t in batch]
            t_sample,t_generate = teacher(all_tensor)
            logits,generate = model(input_tensor)
            topv, topi = logits.topk(1)
            for predic, label in zip(topi.tolist(),target_tensor.tolist()):
                if predic[0] == label:
                    correct+=1
                count+=1
            # print(t_sample[0])
            # print(target_tensor[0])
            # print([round(i.item(),4) for i in F.softmax(t_sample/24,dim=1)[0]])
            loss1 = criterion1(logits, target_tensor)
            loss2 = criterion2(generate, t_generate)
            loss3 = criterion3(logits,F.softmax(t_sample/24,dim=1))

            optimizer.zero_grad()
            loss = 0.1*loss1 + 0.9*loss3
            train_ls_1 += loss.item()
            loss.backward()
            optimizer.step()
        # t_acc = get_acc(dataset,model)
        v_acc = get_acc(valset,model)
        if v_acc > max_acc :
            max_acc = v_acc
            torch.save(model.state_dict(), 'task3_student_3'+'.pkl')
        print('Epoch: {}'.format(epoch),'    '\
            ,'train loss: {:.4}'.format(train_ls_1/(step+1)),'\t'\
            ,'train acc: {:.4}'.format(correct/count)
            ,'val acc: {:.4}'.format(v_acc),'\t'\
            # ,'{:.4}'.format(model.weight[0].item()/weight_sum)
            # ,'{:.4}'.format(model.weight[1].item()/weight_sum),'\t'\
            ,'{:.4}'.format(loss1.item())
            ,'{:.4}'.format(loss2.item())
            ,'{:.4}'.format(loss3.item())
            )
        chart_data['epoch'].append(epoch)
        chart_data['tarin_loss'].append(train_ls_1/(step+1))
        chart_data['train_acc'].append(correct/count)
        chart_data['val_acc'].append(v_acc)
        draw_chart(chart_data,'task3_student_3')