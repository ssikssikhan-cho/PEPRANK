# Copyright 2024 Joon-Sang Park. All Rights Reserved.

import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import numpy as np


class GAT(nn.Module):
    def __init__(self, n_features, n_heads = 2):
        super(GAT, self).__init__()
        self.nheads = n_heads
        self.W = nn.Linear(n_features, n_features, bias = False) if 1 == n_heads else nn.Parameter(torch.zeros(n_heads, 1, n_features, n_features))
        #self.LN = nn.LayerNorm(n_features)
        #self.actiW = nn.LeakyReLU()
        self.E = nn.Parameter(torch.zeros((n_features, n_features))) if 1==n_heads else nn.Parameter(torch.zeros((n_heads, 1, n_features, n_features)))
        #self.E = nn.Linear(n_features, n_features, bias = False) #jsp 2024-10-15T21:09.pth.tar
        #self.gate = nn.Linear(n_features * 2, 1)
        self.gate1 = nn.Linear(n_features, 1, bias=False)
        self.gate2 = nn.Linear(n_features, 1)

    def forward(self, x, adj):
        if 1==self.nheads:
            h = self.W(x)
            #h = self.actiW(self.W(x))
            e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h, self.E), h))
            e = e + e.permute((0, 2, 1))
            attention = torch.where(adj > 0, e, -9e15)
            attention = F.softmax(attention, dim=1)
            h = F.relu(torch.einsum('aij,ajk->aik', (attention * adj, h)))
        else:
            h = torch.matmul(x, self.W)
            #h = self.actiW(torch.matmul(x, self.W))
            e = torch.einsum('imjl,imkl->imjk', (torch.matmul(h, self.E), h))
            e = e + e.permute((0, 1, 3, 2))
            attention = torch.where(adj > 0, e, -1e25)
            attention = F.softmax(attention, dim=2)
            h = F.relu(torch.mean(torch.einsum('abij,abjk->abik', (attention * adj, h)), dim=0))
        #coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(1, 1, x.size(-1))##calculate z_i
        coeff = torch.sigmoid(self.gate1(x) + self.gate2(h))
        retval = coeff * x + (1 - coeff) * h
        return retval


class mlGAT(nn.Module):
    def __init__(self, n_features, n_heads = 3, n_layers = 4, dropout = 0.3):
        super(mlGAT, self).__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.zeros((n_heads, 1, n_features, n_features))) for i in range(n_layers)])
        #self.actW = nn.LeakyReLU()
        self.E = nn.ParameterList([nn.Parameter(torch.zeros((n_heads, 1, n_features, n_features))) for i in range(n_layers)])
        self.actA = nn.ReLU()
        self.gate1 = nn.Linear(n_features, 1, bias=False)
        self.gate2 = nn.Linear(n_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        for k in range(len(self.W)):
            h = torch.matmul(x, self.W[k])
            #h = self.actW(h)
            e = torch.einsum('imjl,imkl->imjk', (torch.matmul(h, self.E[k]), h))
            e = e + e.permute((0, 1, 3, 2))
            attention = torch.where(adj > 0, e, -1e25)
            attention = F.softmax(attention, dim=2)
            h = torch.einsum('abij,abjk->abik', (attention * adj, h))
            h = self.dropout(self.actA(h)) if k < len(self.W) - 1 else self.actA(torch.mean(h, dim=0))
        coeff = torch.sigmoid(self.gate1(x) + self.gate2(h))
        retval = coeff * x + (1 - coeff) * h
        return retval
        
class GNN_EA(nn.Module):
    def __init__(self, n_atom_features = 35, n_heads = 3, n_gat_layers = 4, dim_gat_feat = 70, 
                 dim_fcl_feat = 64, n_fcl = 4, dropout = 0.3, mu_init = 0., dev_init = 1.,
                 int_cutoff = 10):
        super(GNN_EA, self).__init__()

        self.cutoff = int_cutoff
        self.nheads = n_heads
        self.featem = nn.Linear(n_atom_features, dim_gat_feat, bias=False)
        if (n_heads == 1):
            self.gatlayer = nn.ModuleList([GAT(dim_gat_feat, n_heads=1) for i in range(n_gat_layers)])
        else:
            self.gatlayer = mlGAT(dim_gat_feat, n_heads=n_heads, n_layers = n_gat_layers, dropout=dropout) 

        self.FC = nn.ModuleList([nn.Linear(dim_gat_feat, dim_fcl_feat) if i== 0 else
                                 nn.Linear(dim_fcl_feat, 1) if i == n_fcl - 1 else
                                 nn.Linear(dim_fcl_feat, dim_fcl_feat) for i in range(n_fcl)])

        self.mu = nn.Parameter(torch.Tensor([mu_init]))
        self.dev = nn.Parameter(torch.Tensor([dev_init]))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,data,device):
        H, A1, A2, V, num_atoms = data
        H = self.featem(H)
        A2 = torch.where(A2<=self.cutoff, torch.exp(-torch.pow(A2-self.mu, 2)/self.dev), 0) + A1
        if (self.nheads == 1):
            for k in range(len(self.gatlayer)):
                H = self.gatlayer[k](H, A2) - self.gatlayer[k](H, A1)
                H = self.dropout(H)
        else:
            H2 = self.gatlayer(H, A2) 
            H = H2 - self.gatlayer(H, A1)

        #c_hs = c_hs*c_valid.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
        #c_hs = c_hs.sum(1)
        H = (H*V.unsqueeze(-1)).sum(1)
        #old_c_hs = c_hs
        #batch_size = len(num_atoms)
        #c_hs = torch.empty(batch_size, c_hs.shape[2]).to(device)
        #for batch_idx in range(batch_size):
        #    natoms = int(num_atoms[batch_idx])
        #    c_hs[batch_idx]=old_c_hs[batch_idx,:natoms].sum(0)

        for k in range(len(self.FC)):
            H = self.FC[k](H)
            H = F.relu(self.dropout(H)) if len(self.FC) -1 > k else torch.sigmoid(H)

        return H.view(-1)

    
class npzdataset(Dataset):
    def __init__(self,file_list):
        self.filelist = file_list
    def __getitem__(self, idx):
        file_path=self.filelist[idx]
        data=np.load(file_path)
        return data
    def __len__(self):
        return len(self.filelist)
