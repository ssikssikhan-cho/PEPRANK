#!/usr/bin/env python3

import os
import pandas as pd
from multiprocessing import Pool

os.chdir("/workspace/DeepLT/Reranking/Pep_Pro")


def runDockQ(pdb_fn_ref):
    pdb_fn, ref_fn, index, total_files = pdb_fn_ref
    best_dockq = None
    cmd = f"DockQ {pdb_fn} {ref_fn} --short"

    lines = os.popen(cmd).readlines()
    if len(lines) == 0:
        print(f"Processing file {index + 1} of {total_files}: {pdb_fn} - Result: {best_dockq}")
        return (pdb_fn, ref_fn, None)
    
    for line in lines:
        if line.startswith('DockQ'):
            parts = line.split()
            dockq_score = float(parts[1])  # DockQ 점수
            if best_dockq is None or dockq_score > best_dockq:
                best_dockq = dockq_score  # 가장 높은 DockQ 점수 업데이트
    
    print(f"Processing file {index + 1} of {total_files}: {pdb_fn} - Result: {best_dockq}")
    return (pdb_fn, ref_fn, best_dockq)  # 가장 높은 DockQ 점수 반환


# 데이터셋 경로 설정
negative_selected_dataset = 'negative_selected_dataset'
negative_dockQ_needed_dataset = 'negative_dockQ_needed_dataset'
output_csv = 'dockQ_results.csv'

# 결과를 저장할 리스트
results = []

# negative_dockQ_needed_dataset 디렉토리 내의 PDB 파일 읽기
pdb_files = [f for f in os.listdir(negative_dockQ_needed_dataset) if f.endswith('.pdb')]
total_files = len(pdb_files)

# PDB 파일과 원본 PDB 파일 경로 쌍 생성
pdb_ref_pairs = []
for index, pdb_file in enumerate(pdb_files):
    native_pdb_file = pdb_file[:-6] + ".pdb"
    native_pdb_path = os.path.join(negative_selected_dataset, native_pdb_file)

    if os.path.isfile(native_pdb_path):
        pdb_ref_pairs.append((os.path.join(negative_dockQ_needed_dataset, pdb_file), native_pdb_path, index, total_files))

# 멀티 프로세싱을 이용한 DockQ 계산
with Pool() as pool:
    results = pool.map(runDockQ, pdb_ref_pairs)

# 결과를 DataFrame으로 변환
df = pd.DataFrame(results, columns=['DockQ_PDB', 'Native_PDB', 'best_dockQ'])

# CSV 파일로 저장
df.to_csv(output_csv, index=False)

print(f"DockQ 결과가 {output_csv} 파일에 저장되었습니다.")





"""
#!/usr/bin/env python3

import os
import pandas as pd


os.chdir("/workspace/DeepLT/Reranking/Pep_Pro")


def runDockQ(pdb_fn, ref_fn):
    best_dockq = None
    cmd = f"DockQ {pdb_fn} {ref_fn} --short"

    lines = os.popen(cmd).readlines()
    if len(lines) == 0:
        print(best_dockq)
        return None
    
    for line in lines:
        if line.startswith('DockQ'):
            parts = line.split()
            dockq_score = float(parts[1])  # DockQ 점수
            if best_dockq is None or dockq_score > best_dockq:
                best_dockq = dockq_score  # 가장 높은 DockQ 점수 업데이트
    
    print(best_dockq)
    return best_dockq  # 가장 높은 DockQ 점수 반환



# 데이터셋 경로 설정
negative_selected_dataset = 'negative_selected_dataset'
negative_dockQ_needed_dataset = 'negative_dockQ_needed_dataset'
output_csv = 'dockQ_results.csv'

# 결과를 저장할 리스트
results = []

pdb_files = [f for f in os.listdir(negative_dockQ_needed_dataset) if f.endswith('.pdb')]
total_files = len(pdb_files)

# negative_dockQ_needed_dataset 디렉토리 내의 PDB 파일 읽기
for index, pdb_file in enumerate(pdb_files):
    if pdb_file.endswith('.pdb'):
        # 원본 PDB 파일 이름 생성
        native_pdb_file = pdb_file[:-6] + ".pdb"
        
        # 원본 PDB 파일 경로
        native_pdb_path = os.path.join(negative_selected_dataset, native_pdb_file)
        
        # DockQ 값 계산
        if os.path.isfile(native_pdb_path):
            best_dockQ = runDockQ(os.path.join(negative_dockQ_needed_dataset, pdb_file), native_pdb_path)
            results.append((pdb_file, native_pdb_file, best_dockQ))
    
    print(f"Processing file {index + 1} of {total_files}: {pdb_file}")

# 결과를 DataFrame으로 변환
df = pd.DataFrame(results, columns=['DockQ_PDB', 'Native_PDB', 'best_dockQ'])

# CSV 파일로 저장
df.to_csv(output_csv, index=False)

print(f"DockQ 결과가 {output_csv} 파일에 저장되었습니다.")
"""