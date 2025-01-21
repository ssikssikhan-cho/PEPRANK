# Copyright 2024 Joon-Sang Park. All Rights Reserved.

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from commonfncs import get_atom_feature
from scipy.spatial import distance_matrix
from Bio.PDB import *

import biographs as bg
import networkx as nx

def collate_fn_orgA2(batch):
    max_natoms = max([len(item['H']) for item in batch if item is not None])
    #for item in batch:
    #    print('H shape: ', item['H'].shape) 
    Hs = [len(item['H']) for item in batch if item is not None]
    #print('Hs: ', Hs)
    #print('len(batch): ', len(batch))
    #max_nresidues = max([len(item['H']) for item in batch if item is not None])
    #print('max_nresidues: ', max_nresidues)
    H = np.zeros((len(batch), max_natoms, batch[0]['H'].shape[1]))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))
    V = np.zeros((len(batch), max_natoms))
    Y = np.zeros((len(batch), 1))
    Atoms_Number=[]
    #Residues_Number=[]
    for i in range(len(batch)):
        natom = len(batch[i]['H'])
        H[i, :natom] = batch[i]['H']
        A1[i, :natom, :natom] = batch[i]['A1']
        A2[i, :natom, :natom] = batch[i]['A2'] 
        V[i, :natom] = batch[i]['V']
        Y[i] = batch[i]['Y']
        Atoms_Number.append(natom)
    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    V = torch.from_numpy(V).float()
    #Y = np.array(Y)
    Y = np.ravel(Y)
    Y = torch.from_numpy(Y).float()
    Atoms_Number=torch.Tensor(Atoms_Number)

    return H, A1, A2, V, Atoms_Number, Y 

def collate_genA2(batch):
    max_natoms = max([len(item['H']) for item in batch if item is not None])
    #for item in batch:
    #    print('H shape: ', item['H'].shape) 
    Hs = [len(item['H']) for item in batch if item is not None]
    #print('Hs: ', Hs)
    #print('len(batch): ', len(batch))
    #max_nresidues = max([len(item['H']) for item in batch if item is not None])
    #print('max_nresidues: ', max_nresidues)
    H = np.zeros((len(batch), max_natoms, batch[0]['H'].shape[1]))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))
    V = np.zeros((len(batch), max_natoms))
    #Y = np.zeros((len(batch), 1))
    Y = []
    Atoms_Number=[]
    #Residues_Number=[]
    for i in range(len(batch)):
        natom = len(batch[i]['H'])
        #print('natom: ', natom)
        H[i, :natom] = batch[i]['H']
        #H[i, :nresidue] = batch[i]['H']
        A1[i, :natom, :natom] = batch[i]['A1']
        #A2[i, :natom, :natom] = batch[i]['A2'] #jsp replaceed with below
        A2[i, :natom, :natom] = np.where(batch[i]['A2']<=10,np.exp(-np.power(batch[i]['A2'], 2)),np.zeros_like(batch[i]['A2']))
        A2[i, :natom, :natom] += batch[i]['A1'] 
        V[i, :natom] = batch[i]['V']
        Y.append(batch[i]['Y'])
        #print('i: ', i)
        #print('nresidue: ', nresidue)
        #print('batch[i][A1]: ', batch[i]['A1'].shape)
        Atoms_Number.append(natom)
    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    V = torch.from_numpy(V).float()
    Y = np.array(Y)
    Y = np.ravel(Y)
    Y = torch.from_numpy(Y).float()
    Atoms_Number=torch.Tensor(Atoms_Number)

    return H, A1, A2, V, Atoms_Number, Y 

