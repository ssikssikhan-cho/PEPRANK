# Copyright Joon-Sang Park

import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import numpy as np

int_cutoff = 10

class GAT(nn.Module):
    def __init__(self, n_features, n_out_feature):
        super(GAT, self).__init__()
        self.W = nn.Linear(n_features, n_out_feature, bias = False)
        #self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        #self.A = nn.Linear(n_out_feature, n_out_feature, bias = False) #jsp 2024-10-15T21:09.pth.tar
        #self.gate = nn.Linear(n_out_feature * 2, 1)
        self.gate1 = nn.Linear(n_out_feature, 1, bias=False)
        self.gate2 = nn.Linear(n_out_feature, 1)

    def forward(self, x, adj):
        h = self.W(x)#x'=W*x_in
        batch_size = h.size()[0]
        N = h.size()[1]#num_atoms
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h, self.A), h))#A is E in the paper,
        #This function provides a way of computing multilinear expressions (i.e. sums of products) using the Einstein summation convention.
        e = e + e.permute((0, 2, 1))
        zero_vec = -9e15 #* torch.ones_like(e) 
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #output_attention=attention
        #attention = attention * adj#final attention a_ij
        h_prime = F.relu(torch.einsum('aij,ajk->aik', (attention * adj, h)))#x'' in the paper
        #coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(1, 1, x.size(-1))##calculate z_i
        coeff = torch.sigmoid(self.gate1(x) + self.gate2(h_prime))
        retval = coeff * x + (1 - coeff) * h_prime#final output,linear combination
        return retval
        #if request_attention:
        #    return output_attention,retval
        #else:
        #    return retval

        
class GNN_EA(nn.Module):
    def __init__(self, params, n_atom_features = 56, n_gat_layers = 4):
        super(GNN_EA, self).__init__()
        d_graph_layer = 70#params['d_graph_layer']
        d_FC_layer = params['d_FC_layer']
        n_FC_layer = params['n_FC_layer']
        self.dropout_rate = params['do_rate']
        self.dropout = nn.Dropout(self.dropout_rate)

        self.layers1 = [d_graph_layer for i in range(n_gat_layers+1)]
        self.gconv1 = nn.ModuleList \
            ([GAT(self.layers1[i], self.layers1[ i +1]) for i in range(len(self.layers1) -1)])

        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i== 0 else
                                 nn.Linear(d_FC_layer, 1) if i == n_FC_layer - 1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])

        self.mu = nn.Parameter(torch.Tensor([params['mu']]).float())
        self.dev = nn.Parameter(torch.Tensor([params['dev']]).float())
        self.featem = nn.Linear(n_atom_features, d_graph_layer, bias=False)
        #self.params=params
    
    def forward(self,data,device):
        H, A1, A2, V, num_atoms = data
        H = self.featem(H)
        A2 = torch.where(A2<=10, torch.exp(-torch.pow(A2-self.mu, 2)/self.dev), 0) + A1
        for k in range(len(self.gconv1)):
            H = self.gconv1[k](H, A2) - self.gconv1[k](H, A1)
            H = F.dropout(H, p=self.dropout_rate, training=self.training)

        #c_hs = c_hs*c_valid.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
        #c_hs = c_hs.sum(1)
        H = (H*V.unsqueeze(-1)).sum(1)
        #old_c_hs = c_hs
        #batch_size = len(num_atoms)
        #c_hs = torch.empty(batch_size, c_hs.shape[2]).to(device)
        #for batch_idx in range(batch_size):
        #    natoms = int(num_atoms[batch_idx])
        #    c_hs[batch_idx]=old_c_hs[batch_idx,:natoms].sum(0)

        #c_hs = self.fully_connected(c_hs)
        for k in range(len(self.FC)-1):
            H = self.FC[k](H)
            H = F.dropout(H, p=self.dropout_rate, training=self.training)
            H = F.relu(H)

        H = self.FC[len(self.FC)-1](H)
        H = torch.sigmoid(H)

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