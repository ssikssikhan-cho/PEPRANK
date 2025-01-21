# 컨테이너 밖에서 실행한 코드고 python3로 실행해야 함
# 해당하는 경로의 데이터셋 전체에서 필요한 부분만 빼오기 위해서 사용한 코드임
"""
# -*- coding: utf-8 -*- 

import glob
import os
import shutil

# 원본 경로들
path1 = '/mnt/rv1/althome/jhhwang/gpd-bin-org/gpd_decoys/*/????/model'
path2 = '/mnt/rv1/althome/jhhwang/gpd-bin-org/gpd_decoys/*/model'

# 파일 경로들을 glob을 사용하여 찾기
model_files = glob.glob(os.path.join(path1, 'model_?.pdb')) + glob.glob(os.path.join(path2, 'model_?.pdb'))

# 복사할 대상 디렉토리의 기본 경로
destination_base = os.path.expanduser('~/negative_dockQ_needed_dataset')

# 각 파일에 대해 복사 작업 수행
for file_path in model_files:
    # 경로를 나누고, gpd_decoys 디렉토리의 이름 추출
    parts = file_path.split('/')
    
    # gpd_decoys 하위 디렉토리 이름 추출 (첫 번째 경로의 경우: 네번째 요소, 두 번째 경로의 경우: 세번째 요소)
    for i, part in enumerate(parts):
        if 'gpd_decoys' in parts[i-1]:
            identifier = part
            break
    
    # 대상 디렉토리 생성
    destination_dir = os.path.join(destination_base, identifier)
    os.makedirs(destination_dir, exist_ok=True)
    
    # 파일 복사
    destination_file = os.path.join(destination_dir, os.path.basename(file_path))
    shutil.copy(file_path, destination_file)
    print(f"Copied {file_path} to {destination_file}")
"""


# 옮기고 나니까 positive_dataset이랑 구조가 다른 게 불편해서 똑같도록 맞춰줄 거임
import os
import shutil
os.chdir("/workspace/DeepLT/Reranking/Pep_Pro")

# negative_dataset 디렉토리 경로 설정
root_dir = 'negative_dockQ_needed_dataset'

# 모든 하위 디렉토리 탐색
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # 파일명이 'model_'로 시작하고 '.pdb'로 끝나는 파일만 처리
        if file.startswith('model_') and file.endswith('.pdb'):
            # 파일 경로 생성
            old_file_path = os.path.join(subdir, file)
            
            # 상위 디렉토리 이름 가져오기
            dir_name = os.path.basename(subdir)
            
            # 파일 번호 추출 (예: 'model_1.pdb'에서 '1' 추출)
            file_num = file.split('_')[1].split('.')[0]
            
            # 새로운 파일 이름 생성 (예: '1A0N_B_14_1.pdb')
            new_file_name = f"{dir_name}_{file_num}.pdb"
            
            # 새로운 파일 경로 생성
            new_file_path = os.path.join(root_dir, new_file_name)
            
            # 파일 이동 및 이름 변경
            shutil.move(old_file_path, new_file_path)
            print(f"Renamed '{old_file_path}' to '{new_file_path}'")
    
    # 기존 디렉토리 삭제
    if not os.listdir(subdir):
        os.rmdir(subdir)
        print(f"Deleted empty directory '{subdir}'")



# 컨테이너 밖에서 실행한 코드고 python3로 실행해야 함
# dockQ 사용하려면 원본 pdb 파일도 받아와야 해서 만든 코드임
"""
# -*- coding: utf-8 -*- 
import glob
import os
import shutil

# 원본 데이터 경로 설정
source_base = '/mnt/rv1/althome/jhhwang/gpd-bin-org/gpd_decoys'

# 모든 서브디렉토리 검색 (예: 1LWU_B_4)
subdirs = glob.glob(os.path.join(source_base, '*_?_*'))

# 각 서브디렉토리에서 pdb 파일 추출 및 복사
for subdir in subdirs:
    # 디렉토리 이름에서 '_?_' 부분 제거하여 pdb 파일 이름 생성
    subdir_name = os.path.basename(subdir)
    pdb_name = subdir_name[:4] + '.pdb'

    # 첫 번째 위치: 직접 해당 디렉토리 안에서 파일 찾기
    pdb_path = os.path.join(subdir, pdb_name)

    # 두 번째 위치: 서브디렉토리 (예: 1LWU_B_4/1LWU)에서 파일 찾기
    pdb_subdir_path = os.path.join(subdir, subdir_name[:4], pdb_name)

    # 파일이 존재하는지 확인하고 복사
    if os.path.exists(pdb_path):
        destination_path = os.path.expanduser('~/negative_not_selected_dataset/' + pdb_name)
        shutil.copy(pdb_path, destination_path)
        print("Copied {} to {}".format(pdb_path, destination_path))
    elif os.path.exists(pdb_subdir_path):
        destination_path = os.path.expanduser('~/negative_not_selected_dataset/' + pdb_name)
        shutil.copy(pdb_subdir_path, destination_path)
        print("Copied {} to {}".format(pdb_subdir_path, destination_path))
"""



# 컨테이너 밖에서 실행한 코드고 python3로 실행해야 함
# negative dataset이 부족해서 다음 경로에서 받아오는 코드임
"""
# -*- coding: utf-8 -*- 
import os
import shutil
import glob

# 원본 파일 경로와 대상 폴더 경로 설정
source_path = '/mnt/rv1/althome/jhlee/MEGADOCK_output/*/*.out.decoy.*.pdb'
target_folder = os.path.expanduser('~/negativeda_dataset')

# 대상 폴더가 존재하지 않으면 생성
os.makedirs(target_folder, exist_ok=True)

# 파일 검색 및 복사
for file in glob.glob(source_path):
    if file.endswith(tuple(f"{i}.pdb" for i in range(995, 1001))):
        shutil.copy(file, target_folder)

print("파일 복사가 완료되었습니다.")

"""