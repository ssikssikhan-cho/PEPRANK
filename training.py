#-*- coding:utf-8 -*-

# nohup python DeepLT/Reranking/Pep_Pro/training.py -F F --mode 3 --fold 5 &
# tail -f nohup.out

import os
import sys
os.chdir("/workspace/DeepLT/Reranking/Pep_Pro")
sys.path.append("../GNN_DOVE")

from ops.argparser import argparser
from model.GNN_Model import GNN_Model
import torch
from prepare_learning_data_a import collate_fn, Data_Sampler, Dockground_Dataset
from data_processing.Prepare_Input import Prepare_Input
from ops.train_utils import count_parameters,initialize_model
from data_processing.Single_Dataset import Single_Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from train_GNN import train_GNN
import time



# modeling
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
params = argparser()
model = GNN_Model(params)
device = torch.device('cuda') # cuda:1은 DEVICE 2임
model.to(device)
print("Device:",device)
print("Curr: ",torch.cuda.current_device())

print(model)
# parameters
params['gpu'] = 1, 2, 3
learning_rate = 0.0002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()
root_path = '/workspace/DeepLT/Reranking/Pep_Pro/'
print(params)

list_posfile = []
list_negfile = []
    
train_path_correct = root_path +  'positive_dataset/npz/'
list_posfile += [train_path_correct + x for x in os.listdir(train_path_correct) if ".npz" in x]

train_path_incorrect = root_path +  'negative_dataset/npz/'
list_negfile += [train_path_incorrect + x for x in os.listdir(train_path_incorrect) if ".npz" in x]

print("correct ",len(list_posfile))
print("incorrect",len(list_negfile))
train_list=[]


for i in range(int(len(list_negfile) / len(list_posfile))):
    train_list += list_posfile

train_list+=list_negfile

loss_list = []
best_acc = 0
epoch = 50

for k in range(epoch):
    print(k)
    start = time.time()
    train_dataset = Single_Dataset(train_list) 
    train_dataloader = DataLoader(train_dataset, 2, shuffle=True,
                                  num_workers=params['num_workers'], collate_fn=collate_fn)
    train_loss, train_Accu= train_GNN(model, train_dataloader, optimizer, loss_fn, device)
    
    print("Avg loss, Avg Accu",train_loss, train_Accu)
    end = time.time()
    print('time', end-start)
    state = {
        'epoch': k + 1,
        'state_dict': model.state_dict(),
        'loss': train_loss,
        'best_roc': best_acc, 
        'optimizer': optimizer.state_dict(),
    }

import datetime
now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d_%H:%M:%S')
nowDatetime = str(nowDatetime)
#print(loss_list)
PATH = root_path + 'model/8_25.pth.tar'
torch.save(model.state_dict(), PATH)
