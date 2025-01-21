# Copyright 2024 Joon-Sang Park. All Rights Reserved.

import os
import shutil
import numpy as np
from generate_npz import generate_npz_file
#from training_preparation import generate_npz_file
from ea_gnn import GNN_EA, npzdataset
import torch
from collatefncs import collate_fn_orgA2
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn as nn
from argparser import argparser
from commonfncs import getatomfeaturelen, getatomfeaturelen_f

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
    model_path = params['parampath']
    #model_path = "model/2024-10-15T21:09.pth.tar"
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
        input_file = generate_npz_file(input_pdb_path, npzdirpf='npz-nf', newfeat = True, forcenpzgen=True)
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



if __name__ == "__main__":
    params = argparser()
    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
    inference_dir(params)






