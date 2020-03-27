import torch
import time
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
        output = model(features = input_tensor)
        topv, topi = output.topk(1)
        for label,predic in zip(input_id.tolist(),topi.tolist()):
            csv_writer.writerow([label,testset.label_id[predic[0]]])
    csv_file.close()
def get_acc(dataset,model):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    count = 0
    correct = 0
    for step, (batch) in enumerate(loader):
        input_tensor,target_tensor=[t.to(device) for t in batch]
        output = model(features = input_tensor)
        topv, topi = output.topk(1)
        for predic, label in zip(topi.tolist(),target_tensor.tolist()):
            if predic[0] == label:
                correct+=1
            count+=1
    return correct/count
class Dataset(Dataset):
    def __init__(self, file, train=True):
        self.train = train
        self.data = pd.read_csv(file, encoding='utf-8')
        self.label_id = ['A','B','C','D','E','F','G','H','I','W','X','Y']
        self.len = len(self.data)
    def __getitem__(self, index):
        if self.train:
            feature = self.data.iloc[index,1:-1].values.astype(np.float32)
            category = self.data.iloc[index,-1]
            try:
                label_tensor = self.label_id.index(category)
            except:
                print(self.data.iloc[index].values)
            feature_tensor = torch.tensor(feature)
            mask = torch.Tensor([1]*len(feature))
            mask[0] = 0
            # for i in range(random.randint(0,3)):
            #     mask[random.randint(0,len(mask)-1)] = 0
            feature_tensor = feature_tensor*mask
            return feature_tensor,label_tensor
        else:
            input_id = self.data.iloc[index,0]
            feature = self.data.iloc[index,1:].values.astype(np.float32)
            feature = np.nan_to_num(feature)
            return input_id,feature
    def __len__(self):
        return self.len
class Model(nn.Module):
    def __init__(self, input_size,hidden_size,category_size):
        super(Model, self).__init__()
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
        self.fc3 =  nn.Sequential(
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
        logits = self.fc3(logits)
        logits = self.classifier(logits)
        logits = self.softmax(logits)
        return logits

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    batch_size = 256
    hidden_size = 64
    drop_pro = 0.2
    learning_rate = 0.002

    torch.manual_seed(1010)
    dataset = Dataset('train.csv')
    trainset, valset  = data.random_split(dataset, (int(len(dataset)*0.8) ,dataset.len-int(len(dataset)*0.8)))
    train_loader = DataLoader(trainset, batch_size=batch_size)

    model = Model(10, hidden_size, len(dataset.label_id))
    model.load_state_dict(torch.load('model_task1_3.pkl'))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    print(get_acc(valset,model))
    get_upload(model)
    exit()
    for epoch in range(15000):
        train_ls = 0
        model.train()
        # print(time.asctime( time.localtime(time.time()) ))
        count = 0
        correct = 0
        max_acc = 0
        for step, (batch) in enumerate(train_loader):
            input_tensor,target_tensor = [t.to(device) for t in batch]
            output = model(features = input_tensor)
            topv, topi = output.topk(1)
            for predic, label in zip(topi.tolist(),target_tensor.tolist()):
                if predic[0] == label:
                    correct+=1
                count+=1
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_ls += loss.item()
        v_acc = get_acc(valset,model)
        if v_acc > max_acc :
            max_acc = v_acc
            torch.save(model.state_dict(), 'model_task1_3'+'.pkl')
        print('Epoch: ' + str(epoch) + ' train loss: ' + str(train_ls/(step+1)) +\
              '  train acc: ' + str(correct/count) + '  val acc: ' + str(v_acc))