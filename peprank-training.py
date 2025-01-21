# Copyright 2024 Joon-Sang Park. All Rights Reserved.
# Training 

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import numpy as np
import datetime
from argparser import argparser
from ea_gnn import GNN_EA, npzdataset
from commonfncs import getatomfeaturelen_f
from collatefncs import collate_fn_orgA2

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train_model(model,train_dataloader,optimizer,loss_fn,device):

    Loss = AverageMeter()
    model.train()
    iteration = int(len(train_dataloader))
    t = time.time()
    acc_loss = 0
    optimizer.zero_grad()
    for batch_idx, sample in enumerate(train_dataloader):
        H, A1, A2, V, Atom_count, Y = sample
        batch_size = H.size(0)

        pred = model((H.to(device), A1.to(device), A2.to(device), V.to(device), Atom_count.to(device)), device)
        loss = loss_fn(pred, Y.to(device))
        Loss.update(loss.item(), batch_size)
        #acc_loss = acc_loss + loss # uncomment to "backward" on accumulated losses
        if (acc_loss == 0):
            loss.backward()

        if batch_idx % 4 == 0 or iteration -1 == batch_idx:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value = 1.0)
            if acc_loss != 0:
                acc_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx} ended at ", time.time() - t)

    return Loss.avg 

def verifynpzfiles(filelist):
    for file in filelist:
        data = np.load(file)
        try:
            data['H']
            data['A1']
            data['A2'] 
            data['A1'] 
            data['V']
            data['Y']
        except:
            print(file)
    

if __name__ == "__main__":
    params = argparser()
    print(params)

    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']
#    model = GNN_EA(params, n_atom_features=getatomfeaturelen(True))
    model = GNN_EA(getatomfeaturelen_f(), n_heads=params['n_heads'],
                   n_gat_layers = params['n_gat_layers'], dim_gat_feat = params['n_gat_layers'], 
                   dim_fcl_feat = params['dim_fcl_feat'], n_fcl = params['n_fcl'], 
                   dropout = params['dropout'])
    for param in model.parameters():
        if 1 != param.dim():
            nn.init.xavier_normal_(param)
    model = nn.DataParallel(model)
    device = torch.device('cuda') 
    model.to(device)

    list_posfile = []
    list_negfile = []
    
    maxlen = params['maxfnum']
    if not None == params['F']:
        path = params['F']
        list_posfile += [path + x for x in os.listdir(path) if ".npz" in x and not x.startswith('.')]
    if not None == params['F2']:
        path = params['F2']
        list_negfile += [path + x for x in os.listdir(path) if ".npz" in x and not x.startswith('.')]
    if (len(list_posfile) > maxlen):
        del list_posfile[maxlen:]
    if (len(list_negfile) > maxlen):
        del list_negfile[maxlen:]

    if len(list_posfile) == 0 and len(list_negfile) == 0:
        basedir = '/rv2/biodata'
        train_path_correct = basedir + '/pepcomplexdb-bl/complex/fixed-reduce/npz-eas/'
        list_posfile += [train_path_correct + x for x in os.listdir(train_path_correct) if ".npz" in x and not x.startswith('.')]

        train_path_incorrect = basedir + '/neg-dataset-36723/fixed-reduce/npz-eas/'
        list_negfile += [train_path_incorrect + x for x in os.listdir(train_path_incorrect) if ".npz" in x and not x.startswith('.')]
    
    #file_sizes = []
    #for file in list_posfile:
    #    file_sizes.append(os.path.getsize(file))
    #tdata_posfiles = [x for _,x in sorted(zip(file_sizes,list_posfile), reverse=True)]
    #with open("posfilesize.txt", "w") as fout:
    #    print(*sorted(zip(file_sizes,list_posfile)), sep="\n", file=fout)
    if len(list_posfile) != 0 and len(list_negfile) != 0:
        diff = len(list_posfile) - len(list_negfile)
        # trying to fix unbalanced dataset (assuming lists are sorted differently)
        if diff > 0:
            list_negfile = list_negfile[:diff] + list_negfile
        elif diff < 0:
            list_posfile.extend(list_posfile[diff:])

    if len(list_posfile) == 0 and len(list_negfile) ==0:
        print(f'No file found for training, exiting. {len(list_posfile)} {len(list_negfile)}')
        exit()
    print(f'{len(list_posfile)}+{len(list_negfile)} files being used for training.')

    list_train = list_posfile + list_negfile
    loss_list = []
    best_acc = 0
    n_epoch = 50
    os.system('mkdir chkpts 2> /dev/null')
    os.system('mkdir model 2> /dev/null')
    sdatetime = str(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M'))

    print(model)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1)

    train_dataset = npzdataset(list_train) 
    train_dataloader = DataLoader(train_dataset, params['batch_size'], shuffle=True,
                                    num_workers=params['num_workers'], collate_fn=collate_fn_orgA2)
    print("Epochs and mini-batches: ", n_epoch, int(len(train_dataloader)))
    for k in range(n_epoch):
        starttime = time.time()
        print(f'Epoch {k} started at', datetime.datetime.fromtimestamp(starttime).isoformat())
        train_loss = train_model(model, train_dataloader, optimizer, loss_fn, device)
        scheduler.step()
        
        print("Avg loss: ",train_loss)
        print(f'Epoch {k} laptime:', time.time() - starttime)
        if k % 5 == 0:
            torch.save( {'epoch': k, 'state_dict': model.state_dict(),
                'loss': train_loss, 'best_roc': best_acc, 'optimizer': optimizer.state_dict(),
                }, 'chkpts/model'+sdatetime+'-'+str(k)+'.pt')

    path = f'model/{sdatetime}.pth.tar'
    print(f'Training finished, params saved as {path}.')
    torch.save(model.state_dict(), path)
