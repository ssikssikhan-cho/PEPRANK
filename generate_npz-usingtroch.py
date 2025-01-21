import os
import sys
import shutil
import numpy as np
import torch

from data_processing.Extract_Interface import Extract_Interface
from rdkit.Chem.rdmolfiles import MolFromPDBFile
from data_processing.Feature_Processing import get_atom_feature
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from scipy.spatial import distance_matrix

import multiprocessing as mp

input_path = ""
npzdirpf = ""

def generate_npz_file(structure_path):
    # /workspace/DeepLT/Reranking/Pep_Pro/positive_dataset
    if 0 == len(os.path.split(structure_path)[0]) :
        dirpath = input_path 
        pdb_name = structure_path[:-4]
        structure_path = os.path.join(dirpath, structure_path)
    else:
        dirpath = os.path.split(structure_path)[0]
        pdb_name = os.path.split(structure_path)[1][:-4]
    npz_path = os.path.join(dirpath, npzdirpf, pdb_name + ".npz")
    print('npz_path: ', npz_path)
    
    if os.path.isfile(npz_path):
        print(f'{npz_path} already exists.')
        return npz_path

    # print(pdb_name[5:])
    # pdb  file param
    #interface_path = root_path + '/interface/'
    # Check if .rinterface and .linterface files exist
    if (os.path.isfile(os.path.join(dirpath, pdb_name + ".rinterface")) and
            os.path.isfile(os.path.join(dirpath, pdb_name + ".linterface"))):
        print(f"Reusing {pdb_name}.[r|l]interface files.")
        receptor_path = os.path.join(dirpath, pdb_name + ".rinterface")
        ligand_path = os.path.join(dirpath, pdb_name + ".linterface")
    else:
        receptor_path, ligand_path = Extract_Interface(structure_path)
    #receptor_path, ligand_path = Extract_Interface(interface_path)
    receptor_mol = MolFromPDBFile(receptor_path, sanitize=False)
    ligand_mol = MolFromPDBFile(ligand_path, sanitize=False)

    receptor_count = receptor_mol.GetNumAtoms()
    ligand_count = ligand_mol.GetNumAtoms()

    """
    receptor_feature = get_atom_feature(receptor_mol, is_ligand=False)
    ligand_feature = get_atom_feature(ligand_mol, is_ligand=True)

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
    # node indice for aggregation
    valid = np.zeros((receptor_count + ligand_count,))
    valid[:receptor_count] = 1
    input_file=npz_path
    
    
        # Y를 다루는 방식인데 혹시 나중에 쓸까봐 주석 처리해둠
        #if int(pdb_name[5:]) <= 100:
        #    Y = np.array([0.])
        #    print('Y: ', Y)
        #else:
        #    Y = np.array([1.])
        #    print('Y: ', Y)
        
    
    #Y = np.array([1.])
    #print("Y: ", Y)
    
    print('input_file: ', input_file)
    np.savez(input_file,  H=H, A1=agg_adj1, A2=agg_adj2, V=valid, Y = Y)
    return input_file
    """
    
    # CPU가 너무 느려서 GPU를 사용하기 위해 위 주석 문자열에서 수정한 코드임
    receptor_count = receptor_mol.GetNumAtoms()
    ligand_count = ligand_mol.GetNumAtoms()

    receptor_feature = torch.tensor(get_atom_feature(receptor_mol, is_ligand=False), device=device)
    ligand_feature = torch.tensor(get_atom_feature(ligand_mol, is_ligand=True), device=device)

    # get receptor adj matrix
    c1 = receptor_mol.GetConformers()[0]
    d1 = torch.tensor(np.array(c1.GetPositions()), device=device)
    adj1 = torch.tensor(GetAdjacencyMatrix(receptor_mol) + np.eye(receptor_count), device=device)
    
    # get ligand adj matrix
    c2 = ligand_mol.GetConformers()[0]
    d2 = torch.tensor(np.array(c2.GetPositions()), device=device)
    adj2 = torch.tensor(GetAdjacencyMatrix(ligand_mol) + np.eye(ligand_count), device=device)
    
    # combine analysis
    H = torch.cat([receptor_feature, ligand_feature], 0)
    agg_adj1 = torch.zeros((receptor_count + ligand_count, receptor_count + ligand_count), device=device)
    agg_adj1[:receptor_count, :receptor_count] = adj1
    agg_adj1[receptor_count:, receptor_count:] = adj2  # array without r-l interaction
    
    dm = torch.tensor(distance_matrix(d1.cpu(), d2.cpu()), device=device)

    agg_adj2 = agg_adj1.clone()
    agg_adj2[:receptor_count, receptor_count:] = dm
    agg_adj2[receptor_count:, :receptor_count] = dm.T  # with interaction array

    # node indices for aggregation
    valid = torch.zeros((receptor_count + ligand_count,), device=device)
    valid[:receptor_count] = 1
    
    Y = torch.tensor([1.0], device=device)
    #Y = torch.tensor([0.0], device=device)
    #print("Y:", Y.cpu().numpy())

    # Save the npz file
    np.savez(npz_path, H=H.cpu().numpy(), A1=agg_adj1.cpu().numpy(), A2=agg_adj2.cpu().numpy(), V=valid.cpu().numpy(), Y=Y.cpu().numpy())
    return npz_path
    
    # np.savez() : 여러개의 배열을 1개의 압축되지 않은 *.npz 포맷 파일로 저장하기 (Save several arrays into a single file in uncompressed .npz format)
    # print(input_file)
    

def make_npz(input_path): 
    listfiles = [x for x in os.listdir(input_path) if ".pdb" in x]
    listfiles.sort()
    total_files = len(listfiles)  # 전체 파일 개수
    for index, item in enumerate(listfiles):  # 인덱스와 함께 반복
        pdb_name = item[:-4]

        # 3NJIAB는 에러가 있어서 스킵 함
        if pdb_name == "3NJIAB":
            print(f"Skipping {pdb_name} due to known error.")
            continue
        
        input_pdb_path = os.path.join(input_path, item) 
        
        print(f"\nProcessing {index + 1}/{total_files}: {input_pdb_path}\n")  # 진행 상황 출력

        # npz file create & save npz file to path
        try:
            input_file = prepare_npz(input_pdb_path, 'npz-eas')
        except Exception as e:
            print(f"Error processing {pdb_name}: {e}")
            continue
        
    return input_file

def generate_npzs_mp(input_path, num_procs = 10): 
    listfiles = [x for x in os.listdir(input_path) if ".pdb" in x]
    listfiles.sort()
    total_files = len(listfiles)  # 전체 파일 개수
    for index in range(0, total_files, num_procs):
        lastindex = index+num_procs if (index + num_procs < total_files) else total_files 
        items = listfiles[index:lastindex]

        print(f"Processing {index + 1}/{total_files}: {items}")

        pool = mp.Pool(lastindex - index) 
        pool.map(generate_npz_file, items)
        pool.close()
        pool.join()
        if (num_procs != lastindex - index):
            print("last items:", items)

    return


if __name__ == "__main__":
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.append("../GNN_DOVE")
    device = 'cpu'
    input_path = "/rv2/biodata/pepcomplexdb-bl/complex"
    npzdirpf = 'npz-eas' #expanded atom set
    os.system("mkdir " + os.path.join(input_path, npzdirpf))
    #make_npz(input_path)
    generate_npzs_mp(input_path, 10)