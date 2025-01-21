# Copyright 2024 Joon-Sang Park. All Rights Reserved.

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn as nn

# copied from commonfncs.py

elecneg = {"C": 2.55, "N": 3.04, "O": 3.44, "S":2.58, "F":3.98, "P":2.19, "Cl":3.16, "Br":2.96, "B":2.04, "H":2.2, "Se":2.55, "I":2.05, "Xe":2.1, "Si": 1.9, "Xx":2.5} #Xx set to mean
t_six = [0, 1, 2, 3, 4, 5]

def one_hot_encoding(x, mset):
    x = x if (x in mset) else mset[-1]
    return list(map(lambda s: x == s, mset))

t_atom_dti = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']
t_atom = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'Se', 'I', 'Xe', 'Si', 'Xx']
t_degree = [0, 1, 2, 3, 4, 5]
t_totalnumhs = [0, 1, 2, 3, 4]
t_impval = [0, 1, 2, 3, 4, 5]

def getatomfeaturelen(newfeat = False):
    base = len(t_atom_dti) if (not newfeat) else len(list(elecneg)) + 1 #elect neg
    return len(t_degree) + len(t_totalnumhs) + len(t_impval) + base + 1 #isaromatic

def atom_feature(m, atom_i):
    atom = m.GetAtomWithIdx(atom_i)
    atsym = atom.GetSymbol()
    return np.array(one_hot_encoding(atsym, t_atom_dti) +
                    one_hot_encoding(atom.GetDegree(), t_degree) +
                    one_hot_encoding(atom.GetTotalNumHs(), t_totalnumhs) +
                    one_hot_encoding(atom.GetImplicitValence(), t_impval) +
                    [atom.GetIsAromatic()])
                    # (10, 6, 5, 6, 1) --> total 28

def featurize(m, atom_i):
    atom = m.GetAtomWithIdx(atom_i)
    atsym = atom.GetSymbol()
    return np.array(one_hot_encoding(atsym, t_atom) +
                    one_hot_encoding(atom.GetDegree(), t_degree) +
                    one_hot_encoding(atom.GetTotalNumHs(), t_totalnumhs) +
                    one_hot_encoding(atom.GetImplicitValence(), t_impval) +
                    [atom.GetIsAromatic()] +
                    [elecneg[atsym if (atsym in t_atom) else t_atom[-1]]])
                    # (10+5, 6, 5, 6, 1) + 1 --> total 28+6=34


def getatomfeaturelen_f():
    return len(list(elecneg)) + len(t_six) + len(t_six) + 1 + 1

def featurize_f(m, atom_i):
    atom = m.GetAtomWithIdx(atom_i)
    atsym = atom.GetSymbol()
    atomtype = list(elecneg)
    return np.array(one_hot_encoding(atsym, atomtype) +
                    one_hot_encoding(atom.GetDegree(), t_six) +
                    one_hot_encoding(atom.GetTotalNumHs()+atom.GetImplicitValence(), t_six) +
                    [atom.GetIsAromatic()] +
                    [elecneg[atsym if (atsym in atomtype) else atomtype[-1]]])
                    # 15 + 6 + 6 + 1 + 1 = 29

def get_atom_feature_f(m, dist_ligand = 0):
    n = m.GetNumAtoms()
    atom_feature_len = getatomfeaturelen_f()
    H = np.empty((n, atom_feature_len))
    for i in range(n):
        H[i] = featurize_f(m, i)
    if 0 == dist_ligand:
        return H
    if dist_ligand == 1:
        H = np.concatenate([H, np.zeros((n,atom_feature_len))], 1)
    else:
        H = np.concatenate([np.zeros((n,atom_feature_len)), H], 1)
    return H

def get_atom_feature(m, is_ligand=True, newfeat = False):
    n = m.GetNumAtoms()
    atom_feature_len = getatomfeaturelen(newfeat)
    H = np.empty((n, atom_feature_len))
    for i in range(n):
        H[i] = featurize(m, i) if newfeat else atom_feature(m, i)
    if newfeat:
        return H
    if is_ligand:
        H = np.concatenate([H, np.zeros((n,atom_feature_len))], 1)
    else:
        H = np.concatenate([np.zeros((n,atom_feature_len)), H], 1)
    return H


# copied from collatefncs.py

from scipy.spatial import distance_matrix
from Bio.PDB import *

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
    Y = np.ravel(Y)
    Y = torch.from_numpy(Y).float()
    Atoms_Number=torch.Tensor(Atoms_Number)

    return H, A1, A2, V, Atoms_Number, Y 


# copied from ea_gnn.py
import torch.nn.functional as F
from torch.utils.data import Dataset

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

# copy from generate_npz.py
from rdkit.Chem.rdmolfiles import MolFromPDBFile
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from scipy.spatial import distance_matrix
from extract_interface import *
from commonfncs import get_atom_feature

input_path = ''
gNpzdir = ''
Y = np.array([1.0]) 
recifext = ".rec-if.pdb"
ligifext = ".lig-if.pdb"
gNewfeat = False

def generate_npz_file(structure_path, npzdirpf = None, newfeat = None, forcenpzgen = False):
    global input_path
    global Y
    global gNewfeat

    if None == npzdirpf:
        npzdirpf = gNpzdir
    if None == newfeat:
        newfeat = gNewfeat 
    if ('rec-if' in structure_path):
        structure_path = structure_path.replace('rec-if.', '')
    if 0 == len(os.path.split(structure_path)[0]) :
        dirpath = input_path 
        pdb_name = structure_path[:-4]
        structure_path = os.path.join(dirpath, structure_path)
    else:
        dirpath = os.path.split(structure_path)[0]
        pdb_name = os.path.split(structure_path)[1][:-4]
    npz_path = os.path.join(dirpath, npzdirpf, pdb_name + ".npz")
    #npz_path = os.path.join('/home/jsp/biodata/pepcomplexdb/pos/', npzdirpf, pdb_name + ".npz")
    
    if not forcenpzgen and os.path.isfile(npz_path):
        print(f'{npz_path} already exists.')
        return npz_path

    try:
        # Check if interface files exist
        if (os.path.isfile(os.path.join(dirpath, pdb_name + recifext)) and
                os.path.isfile(os.path.join(dirpath, pdb_name + ligifext))):
            print(f"Reusing {pdb_name[:-3]}.*-if.pdb files.")
            rec_path = os.path.join(dirpath, pdb_name + recifext)
            lig_path = os.path.join(dirpath, pdb_name + ligifext)
        else:
            rec_path, lig_path = extract_interface(structure_path)

        receptor_mol = MolFromPDBFile(rec_path, sanitize=False)
        ligand_mol = MolFromPDBFile(lig_path, sanitize=False)
    except:
        print(f'Error processing {pdb_name}, MolFromPDBFile() failed.')
        return None 
    if receptor_mol is None or ligand_mol is None:
        print(f'Error processing {pdb_name}, MolFromPDBFile() failed.')
        return None

    try:
        receptor_count = receptor_mol.GetNumAtoms()
        ligand_count = ligand_mol.GetNumAtoms()
        receptor_feature = get_atom_feature(receptor_mol, is_ligand=False, newfeat=newfeat)
        ligand_feature = get_atom_feature(ligand_mol, is_ligand=True, newfeat=newfeat)
        #print(receptor_feature.shape, ligand_feature.shape)

        # get receptor adj matrix
        c1 = receptor_mol.GetConformers()[0]
        d1 = np.array(c1.GetPositions())
        adj1 = GetAdjacencyMatrix(receptor_mol) + np.eye(receptor_count)
        # get ligand adj matrix
        c2 = ligand_mol.GetConformers()[0]
        d2 = np.array(c2.GetPositions())
        adj2 = GetAdjacencyMatrix(ligand_mol) + np.eye(ligand_count)
        # combine analysis
        H = np.concatenate([receptor_feature, ligand_feature], 0)
        agg_adj1 = np.zeros((receptor_count + ligand_count, receptor_count + ligand_count))
        agg_adj1[:receptor_count, :receptor_count] = adj1
        agg_adj1[receptor_count:, receptor_count:] = adj2  # array without r-l interaction
        dm = distance_matrix(d1, d2)

        agg_adj2 = np.copy(agg_adj1)
        agg_adj2[:receptor_count, receptor_count:] = np.copy(dm)
        agg_adj2[receptor_count:, :receptor_count] = np.copy(np.transpose(dm))  # with interaction array
        # node indice for aggregation to indicate ligand atoms
        valid = np.zeros((receptor_count + ligand_count,))
        valid[receptor_count:] = 1 
        np.savez(npz_path,  H=H, A1=agg_adj1, A2=agg_adj2, V=valid, Y = Y)
    except Exception as e:
        print(f"Error processing {pdb_name}: {e}")
        return None

    return npz_path


npzdirpf = 'npz-lt'

def inference_dir(params):
    global npzdirpf

    input_path=os.path.abspath(params['F'])
    save_path = os.path.join(os.getcwd(), "inf_results")
    os.system(f'mkdir {save_path} 2> /dev/null')
    save_path = os.path.join(save_path, input_path.split('/')[-1])
    os.system(f'mkdir {save_path} 2> /dev/null')

    # loading the model
    #model_path = os.path.join(os.getcwd(), "best_model_pepcompbl")
    #model_path = os.path.join(model_path, "model_50.pth.tar")
    if (None == params['parampath']):
        model_path = '/home/jsp/pros/peprank/model/2024-10-24T12:47.pth.tar'
    else:
        model_path = params['parampath']
    chkpts = True if 'chkpts' in  model_path else False
    model = GNN_EA(getatomfeaturelen(True))
    #model = GNN_EA(getatomfeaturelen_f(), n_heads=params['n_heads'],
    #               n_gat_layers = params['n_gat_layers'], dim_gat_feat = params['n_gat_layers'], 
    #               dim_fcl_feat = params['dim_fcl_feat'], n_fcl = params['n_fcl'])

    # Check if "module key" is present in the state_dict and/or a checkpoint
    state_dict = torch.load(model_path)
    if not isinstance(model, nn.DataParallel):  
        if chkpts:
            for k, v in state_dict.items():
                if 'state_dic' in k:
                    new_state_dict = OrderedDict()
                    for sk, sv in v.items():
                        name = sk[7:] # remove `module.`
                        new_state_dict[name] = sv
                    state_dict = new_state_dict
                    break
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    listfiles=[x for x in os.listdir(input_path) if ".pdb" in x and not "-if" in x]
    maxlen = params['maxfnum']
    if (len(listfiles) > maxlen):
        del listfiles[maxlen:]
    print(len(listfiles), "files to be evaluated.")
    listfiles.sort()
    Input_File_List=[]
    #os.system("mkdir " + os.path.join(input_path, npzdirpf)+" 2> /dev/null")
    for item in listfiles:
        input_pdb_path=os.path.join(input_path,item)
        input_file = generate_npz_file(input_pdb_path, npzdirpf='npz-nf', newfeat = True, forcenpzgen=False)
        #input_file = generate_npz_file(input_pdb_path, npzdirpf=npzdirpf)
        if None != input_file:
            Input_File_List.append(input_file)
    list_npz = Input_File_List
    dataset = npzdataset(list_npz)
    dataloader = DataLoader(dataset, params['batch_size'], shuffle=False,
                            num_workers=params['num_workers'],
                            drop_last=False, collate_fn=collate_fn_orgA2)

    # prediction
    preds = []
    torch.no_grad()
    for idx, sample in enumerate(dataloader):
        H, A1, A2, V, Atom_count, Y = sample
        pred = model((H.to(device), A1.to(device), A2.to(device), V.to(device), Atom_count.to(device)), device)
        preds += list(pred.detach().cpu().numpy())
    
    pred_path = os.path.join(save_path, 'predictions.txt')
    with open(pred_path, 'w') as file:
        file.write("Input\tScore\n")
        for k in range(len(Input_File_List)):
            file.write(Input_File_List[k].split('/')[-1] + "\t%.4f\n" % preds[k])

    pred_sort_path=os.path.join(save_path,"predictions_sorted.txt")
    os.system("sort -n -k 2 -r "+pred_path+" >"+pred_sort_path)

# copied from argparser.py
import argparse
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-F',type=str,help='path to inference/training target(s)') #Path for decoy dir
    parser.add_argument('-F2',type=str,help='path to inference/training target(s)') #Path for second decoy dir
    parser.add_argument('--gpu',type=str,default='0',help='Choose gpu id, example: \'1,2\'(specify use gpu 1 and 2)')
    parser.add_argument("--batch_size", help="batch_size", type=int, default=16)
    parser.add_argument("--num_workers", help="number of (torch) workers", type=int, default=16)
    parser.add_argument("--n_heads", help="number of attention heads", type=int, default=3)
    parser.add_argument("--n_gat_layers", help="number of GAT layers", type=int, default=4)
    parser.add_argument("--dim_gat_feat", help="GAT feature dim", type=int, default=70)
    parser.add_argument("--n_fcl", help="number of FC layers", type=int, default=4)
    parser.add_argument("--dim_fcl_feat", help="dimension of FC layer", type=int, default=64)
    parser.add_argument("--dropout", help="dropout rate", type=float, default=0.3)
    parser.add_argument('--parampath',help='model param file path for infernce',type=str,default=None)
    parser.add_argument('--maxfnum', help='Max # of files used for eval/train', type=int, default = 100000)
    parser.add_argument('--n_epochs', help='', type=int, default = 40)
    args = parser.parse_args()
    params = vars(args)
    return params


if __name__ == "__main__":
    params = argparser()
    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
    inference_dir(params)






