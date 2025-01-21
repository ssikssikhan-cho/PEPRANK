import os

os.chdir("/workspace/DeepLT/Reranking/DeepRank-GNN-esm")

dataset_path = "../Pep_Pro/dataset"

for filename in os.listdir(dataset_path):
    # 파일의 확장자가 .pdb인지 확인
    if filename.endswith(".pdb"):
        # "AB.pdb"로 끝나는지 확인
        cmd = f"python src/deeprank_gnn/predict.py {dataset_path}/{filename} A B"
        print(cmd)
        os.system(cmd)