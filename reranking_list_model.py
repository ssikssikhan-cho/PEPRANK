import os
import sys


sys.path.append("/workspace/DeepLT/Reranking/GNN_DOVE/ops")

from os_operation import mkdir

# 모델을 바꾸려면 Reranking/GNN_DOVE/predict/predict_model_input.py에서 바꿔줘야 함
def reranking():
    fold_path = f"../../vol/Pep_Pro/peppro_exsitunbound_dataset"
    print(f"평가가 실행됩니다.") 
    os.system(f"python main.py --mode=3 -F={fold_path} --num_workers=2 --batch_size=1 --gpu={gpu} --fold=6")


def move_to_result():
    dove_result_path = "../../vol/Pep_Pro/peppro_exsitunbound_result"
    mkdir(dove_result_path)

    os.system(f"mv ./Predict_Result/Multi_Target/Fold_6_Result/peppro_exsitunbound_dataset {dove_result_path}")



gpu = 0

os.chdir("/workspace/DeepLT/Reranking/GNN_DOVE")
reranking()
move_to_result()
