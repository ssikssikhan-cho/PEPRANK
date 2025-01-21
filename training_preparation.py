# Copyright (C) 2024 Joon-Sang Park

import os
import sys
import shutil
import numpy as np
import torch
from rdkit.Chem.rdmolfiles import MolFromPDBFile
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from scipy.spatial import distance_matrix
import multiprocessing as mp

from extract_interface import *
from commonfncs import get_atom_feature_f


input_path = ''
gNpzdir = ''
Y = np.array([1.0]) 
recifext = ".rec-if.pdb"
ligifext = ".lig-if.pdb"

def generate_npz_file(structure_path, npzdirpf = None, forcenpzgen = False):
    global input_path
    global Y

    if None == npzdirpf:
        npzdirpf = gNpzdir
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
    if not os.path.isdir(os.path.join(dirpath, npzdirpf)):
        os.system(f'mkdir {os.path.join(dirpath, npzdirpf)} 2> /dev/null')
    
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
        receptor_feature = get_atom_feature_f(receptor_mol)
        ligand_feature = get_atom_feature_f(ligand_mol)
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

        

def gen_npzs_dir(dirpath, num_procs = 0): 
    listfiles = [x for x in os.listdir(dirpath) if ".pdb" in x and not "-if" in x]
    listfiles.sort()
    numfiles = len(listfiles)
    if numfiles == 0:
        listfiles = [x for x in os.listdir(dirpath) if "rec-if" in x]
        listfiles.sort()
        numfiles = len(listfiles)
    os.system("mkdir " + os.path.join(dirpath, gNpzdir)+" 2> /dev/null")
    for index, item in enumerate(listfiles):
        print(f"\nProcessing {index + 1}/{numfiles}: {item}\n")
        generate_npz_file(item)

def gen_npzs_dir_mp(dirpath, num_procs = 10): 
    listfiles = [x for x in os.listdir(dirpath) if ".pdb" in x and not "-if" in x]
    listfiles.sort()
    numfiles = len(listfiles) 
    os.system("mkdir " + os.path.join(dirpath, gNpzdir)+" 2> /dev/null")
    if numfiles == 0:
        listfiles = [x for x in os.listdir(dirpath) if "rec-if" in x]
        listfiles.sort()
        numfiles = len(listfiles)
    for index in range(0, numfiles, num_procs):
        lastindex = index+num_procs if (index + num_procs < numfiles) else numfiles 
        items = listfiles[index:lastindex]

        print(f"Processing {index + 1}/{numfiles}: {items}")   

        pool = mp.Pool(lastindex - index) 
        pool.map(generate_npz_file, items)
        pool.close()
        pool.join()
        if (num_procs != lastindex - index):
            print("last items:", items)

def verifynpzfile(file):
    try:
        data = np.load(file)
        data['H']
        data['A1']
        data['A2'] 
        data['A1'] 
        data['V']
        data['Y']
    except Exception as e:
        print(f"Error in {file}: {e}")

def verify_npz_dir_mp(dirpath, num_procs = 10): 
    listfiles = [dirpath + x for x in os.listdir(dirpath) if ".npz" in x]
    numfiles = len(listfiles)  # 전체 파일 개수
    for index in range(0, numfiles, num_procs):
        lastindex = index+num_procs if (index + num_procs < numfiles) else numfiles 
        items = listfiles[index:lastindex]
        pool = mp.Pool(lastindex - index) 
        pool.map(verifynpzfile, items)
        pool.close()
        pool.join()

if __name__ == "__main__":
    gNpzdir = 'npz-eas' #expanded atom set
    if (len(sys.argv) > 1):
        input_path = sys.argv[1]
    if (len(sys.argv) > 2):
        gNpzdir = sys.argv[2]
    if (len(sys.argv) > 3):
        Y = np.empty(1)
        Y[0] = float(sys.argv[3])
    elif ('neg' in input_path): #hack!
        Y = np.array([0.])
    print("Params:", input_path, gNpzdir, Y)
    gen_npzs_dir_mp(input_path, 10)



"""
from extract_interface import extract_interface
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolfiles import MolFromPDBFile

def prepare_npz(structure_path):
    # /home/newuser/ML/GNN_DOVE/train_test/train
    root_path=os.path.split(structure_path)[0]
    # 1a2k_99999
    pdb_name = os.path.split(structure_path)[1][:-4]
    
    # print(pdb_name[5:])2
    # pdb  file param
    receptor_path, ligand_path = extract_interface(structure_path)
    receptor_mol = MolFromPDBFile(receptor_path, sanitize=False)
    ligand_mol = MolFromPDBFile(ligand_path, sanitize=False)
    
    receptor_count = receptor_mol.GetNumAtoms()
    ligand_count = ligand_mol.GetNumAtoms()

    #receptor_feature = get_atom_feature(receptor_mol, is_ligand=False)
    #ligand_feature = get_atom_feature(ligand_mol, is_ligand=True)
    receptor_pmol = bg.Pmolecule(receptor_path)
    ligand_pmol = bg.Pmolecule(ligand_path)
    receptor_network = receptor_pmol.network(weight=False, cutoff= 10)
    ligand_network = ligand_pmol.network(weight=False, cutoff= 10)

    parser = PDBParser()
    receptor_structure = parser.get_structure("rec", receptor_path )
    receptor_residue_feature = get_residue_feature(receptor_structure, is_ligand = False)
    ligand_structure = parser.get_structure("lig", ligand_path)
    ligand_residue_feature = get_residue_feature(ligand_structure, is_ligand = True)
    
    rec_CA = []
    lig_CA = []
    receptor_res_count = 0
    for model in receptor_structure:
      for chain in model:
            #CA_coord = []
            for residue in chain:
                receptor_res_count += 1
                if residue.has_id('CA'):

                    #print(residue)
                    rec_CA.append((residue["CA"].get_coord()))
                else:
                    at_c = []
                    for atom in residue:
                        at_c.append(atom.get_coord())
                    rec_CA.append(np.mean(np.array(at_c), axis = 0))
            #print(CA_coord)
    rec_CA = np.array(rec_CA)
    print('a: ', receptor_res_count)
    ligand_res_count = 0
    for model in ligand_structure:
      for chain in model:
            #CA_coord = []
            for residue in chain:
                ligand_res_count += 1
                if residue.has_id('CA'):
                    #print(residue)
                    lig_CA.append((residue["CA"].get_coord()))
                else:
                    at_c = []
                    for atom in residue:
                        at_c.append(atom.get_coord())
                    lig_CA.append(np.mean(np.array(at_c), axis = 0))

            #print(CA_coord)
    lig_CA = np.array(lig_CA)
    
    # get receptor adj matrix
    #c1 = receptor_mol.GetConformers()[0]               #가능한 모든 분자 구조 리스트 
    #d1 = np.array(c1.GetPositions())          # 모든 원자의 위치를 알려줌
    #adj1 = GetAdjacencyMatrix(receptor_mol) + np.eye(receptor_count)        # 분자의 adjacency matrix  + 주대각선 + 1?
    # get ligand adj matrix
    #c2 = ligand_mol.GetConformers()[0]
    #d2 = np.array(c2.GetPositions())
    #adj2 = GetAdjacencyMatrix(ligand_mol) + np.eye(ligand_count)

    res_adj1 = nx.adjacency_matrix(receptor_network).todense() + np.eye(receptor_res_count)
    res_adj2 = nx.adjacency_matrix(ligand_network).todense() + np.eye(ligand_res_count)

    # combine analysis
   # H = np.concatenate([receptor_feature, ligand_feature], 0)
    H_res = np.concatenate([receptor_residue_feature, ligand_residue_feature], 0)
    #agg_adj1 = np.zeros((receptor_count + ligand_count, receptor_count + ligand_count))
    #agg_adj1[:receptor_count, :receptor_count] = adj1
    #agg_adj1[receptor_count:, receptor_count:] = adj2  # array without r-l interaction
    #dm = distance_matrix(d1, d2)
    dm_r = distance_matrix(rec_CA, lig_CA)
    res_agg_adj1 = np.zeros((receptor_res_count+ligand_res_count, receptor_res_count+ligand_res_count))
    res_agg_adj1[:receptor_res_count, :receptor_res_count] = res_adj1
    res_agg_adj1[receptor_res_count:, receptor_res_count:] = res_adj2
    #agg_adj2 = np.copy(agg_adj1)
    res_agg_adj2 = np.copy(res_agg_adj1)
    #agg_adj2[:receptor_count, receptor_count:] = np.copy(dm)
    res_agg_adj2[:receptor_res_count, receptor_res_count:] = np.copy(dm_r)
    #agg_adj2[receptor_count:, :receptor_count] = np.copy(np.transpose(dm))  # with interaction array
    res_agg_adj2[receptor_res_count:, :receptor_res_count] = np.copy(np.transpose(dm_r))
    
    # node indice for aggregation
    #valid = np.zeros((receptor_count + ligand_count,))
    valid_res = np.zeros((receptor_res_count+ ligand_res_count, ))
    #valid[:receptor_count] = 1
    valid_res[:receptor_res_count] = 1
    # /home/newuser/ML/GNN_DOVE/train_test/train/npz/1a2k_9999.npz
    npz_path = root_path + '/npz2/' + pdb_name + '.npz'
    print(npz_path)
    input_file=npz_path
    if int(pdb_name[5:]) <= 100:
        Y = np.array([0.])
    else:
        Y = np.array([1.])
    # np.savez() : 여러개의 배열을 1개의 압축되지 않은 *.npz 포맷 파일로 저장하기 (Save several arrays into a single file in uncompressed .npz format)
    # print(input_file)
    #np.savez(input_file,  H=H, A1=agg_adj1, A2=agg_adj2, V=valid, Y = Y)
    np.savez(input_file, H = H_res, A1 = res_agg_adj1, A2 = res_agg_adj2,V = valid_res)
    return input_file
"""

