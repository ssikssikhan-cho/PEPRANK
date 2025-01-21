#-*- coding:utf-8 -*-

# nohup python DeepLT/Reranking/Pep_Pro/training.py -F F --mode 3 --fold 5 &
# tail -f nohup.out

import os
import sys
#os.chdir("DeepLT/Reranking/Pep_Pro")
sys.path.append("../GNN_DOVE")

from ops.argparser import argparser
from model.GNN_Model_DPE import GNN_Model_DPE
import torch
from prepare_learning_data_dpe import AverageMeter, collate_fn, Data_Sampler, Dockground_Dataset
from data_processing.Prepare_Input import Prepare_Input
from ops.train_utils import count_parameters,initialize_model
from data_processing.Single_Dataset import Single_Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import time

def train_GNN(model,train_dataloader,optimizer,loss_fn,device):

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Loss = AverageMeter()
    Accu1 = AverageMeter()
    #Accu2 = AverageMeter()                                                                                                                                             
    iteration = int(len(train_dataloader))
    loss_list = []
    t = time.time()
    print("Iters: ", iteration)
    for batch_idx, sample in enumerate(train_dataloader):

        H, A1, A2, V, Atom_count, Y = sample
        batch_size = H.size(0)
        H, A1, A2, Y, V = H.cuda(), A1.cuda(), A2.cuda(), Y.cuda(), V.cuda()

#        pred = model.module.train_model((H, A1, A2, V, Atom_count), device)
        pred = model((H, A1, A2, V, Atom_count), device)

        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])                                                                                             
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value = 1.0)
        #loss_list.append(loss)                                                                                                                                         
        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:
            Loss.update(loss.item(), batch_size)
            Accu1.update(pred, batch_size)
            print("batch end: ",batch_idx, time.time() - t)


    return Loss.avg, Accu1.avg#, loss_list 

# modeling
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
params = argparser()
model = GNN_Model_DPE(params)
model = nn.DataParallel(model)
device = torch.device('cuda') 
model.to(device)
print("Device:",device)

print(model)
# parameters
params['gpu'] = 0, 1
learning_rate = 0.0002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()
print(params)

list_posfile = []
list_negfile = []

basedir = '/lab/wnl/jsp'
train_path_correct = basedir + '/biodata/pepcomplexdb/pos/npz/'
list_posfile += [train_path_correct + x for x in os.listdir(train_path_correct) if ".npz" in x]

train_path_incorrect = basedir + '/biodata/pepcomplexdb/neg/npz/'
list_negfile += [train_path_incorrect + x for x in os.listdir(train_path_incorrect) if ".npz" in x]

print("correct ",len(list_posfile))
print("incorrect",len(list_negfile))

#train_list=[]
#for i in range(int(len(list_negfile) / len(list_posfile))):
#    train_list += list_posfile
#print(len(train_list))
train_list = list_posfile

train_list+=list_negfile

loss_list = []
best_acc = 0
epoch = 50

for k in range(epoch):
    print('epoch:', k)
    start = time.time()
    train_dataset = Single_Dataset(train_list) 
    train_dataloader = DataLoader(train_dataset, 8, shuffle=True,
                                  num_workers=params['num_workers'], collate_fn=collate_fn)
    train_loss, train_Accu= train_GNN(model, train_dataloader, optimizer, loss_fn, device)
    
    print("Avg loss, Avg Accu",train_loss, train_Accu)
    end = time.time()
    print('time', end-start)
    torch.save( {
        'epoch': k + 1,
        'state_dict': model.state_dict(),
        'loss': train_loss,
        'best_roc': best_acc, 
        'optimizer': optimizer.state_dict(),
    }, 'chkpts/model'+str(k+1)+'.pt')

import datetime
now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d_%H:%M:%S')
nowDatetime = str(nowDatetime)
print(loss_list)
PATH = 'model/8_25.pth.tar'
torch.save(model.state_dict(), PATH)
