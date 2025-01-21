import pandas as pd
import os
import shutil

os.chdir("/workspace/DeepLT/Reranking/Pep_Pro")


# 입력 CSV 파일 경로
input_csv_path = '0.49_dockq.csv'
# negative_dataset 폴더 경로
output_directory = 'negative_dataset'

# negative_dataset 폴더가 없으면 생성
os.makedirs(output_directory, exist_ok=True)

# CSV 파일 읽기
df = pd.read_csv(input_csv_path)

# DockQ_PDB 열에서 경로 추출
dockq_pdb_paths = df['DockQ_PDB']

# 각 PDB 파일을 negative_dataset 폴더로 복사
for pdb_path in dockq_pdb_paths:
    # 파일 이름 추출
    file_name = os.path.basename(pdb_path)
    # 복사할 경로 설정
    destination = os.path.join(output_directory, file_name)
    
    # 파일 복사
    shutil.copy(pdb_path, destination)

print('PDB 파일들이 negative_dataset 폴더로 복사되었습니다.')
