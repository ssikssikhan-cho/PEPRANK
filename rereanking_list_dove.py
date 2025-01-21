import os
import sys

sys.path.append("./DeepLT/Reranking/GNN_DOVE/ops")

from os_operation import mkdir

def reranking():
    fold_path = f"../Pep_Pro/dataset"
    print(f"평가가 실행됩니다.") 
    os.system(f"python main.py --mode=1 -F={fold_path} --num_workers=4 --batch_size=1 --gpu={gpu} --fold={model_fold}")


def move_to_result():
    dove_result_path = "../Pep_Pro/result_dove"
    mkdir(dove_result_path)

    os.system(f"mv ./Predict_Result/Multi_Target/Fold_{model_fold}_Result/dataset {dove_result_path}")


model_fold = -1
gpu = 0

os.chdir("/workspace/DeepLT/Reranking/GNN_DOVE")
reranking()
move_to_result()
