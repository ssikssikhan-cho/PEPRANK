import os
import subprocess

os.chdir("/workspace/DeepLT/Reranking/Pep_Pro")


# pep_bind.list 파일 경로
pep_bind_file = 'pep_bind.list'

# 디렉토리 경로 설정
not_selected_dir = 'negative_not_selected_dataset'
output_dir = 'negative_selected_dataset'

# 없을 경우 디렉토리 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# pep_bind.list 파일을 읽기
try:
    listfile = open('pep_bind.list', 'r')
except IOError:
    print("list/output file IO error")
    quit()

# 각 줄을 처리
while True:
    # 한 줄 읽기
    listline = listfile.readline()
    # 더 이상 읽을 줄이 없으면 종료
    if not listline:
        break

    # 3. 공백으로 나누기
    parts = listline.split()

    pdb_id = parts[0]      # 예: 1A0N
    pro_chain = parts[1]     # 예: B
    pep_chain = parts[2]     # 예: A
    number = parts[3]       # 예: 14

    # PDB 파일 경로 설정
    pdb_file_path = os.path.join(not_selected_dir, f"{pdb_id}.pdb")
    output_file_name = f"{pdb_id}_{pro_chain}_{number}.pdb"
    output_file_path = os.path.join(output_dir, output_file_name)

    # PDB 파일이 존재하는지 확인
    if not os.path.isfile(pdb_file_path):
        print(f"파일이 존재하지 않습니다: {pdb_file_path}")
        continue
    
    
    
    # 리셉터 체인 선택
    cmd = f"pdb_selchain -{pro_chain} {pdb_file_path} > {pdb_id}_{pro_chain}.pdb"
    print(cmd)
    os.system(cmd)
    pro = f"{pdb_id}_{pro_chain}.pdb"
    
    
    # 펩타이드 체인 선택
    cmd = f"pdb_selchain -{pep_chain} {pdb_file_path} > {pdb_id}_{pep_chain}.pdb"
    print(cmd)
    os.system(cmd)
    pep = f"{pdb_id}_{pep_chain}.pdb"
    
    
    
    # 두 PDB 파일 합치기
    cmd = f"pdb_merge {pro} {pep} > natmp.pdb"
    print(f"Executing: {cmd}")
    os.system(cmd)

    # 합쳐진 PDB 파일을 정리하여 최종 파일로 저장
    cmd = f"pdb_tidy natmp.pdb > {output_file_path}"
    print(f"Executing: {cmd}")
    os.system(cmd)
    
    # 중간에 생성된 PDB 파일 삭제
    intermediate_files = [f"{pdb_id}_{pro_chain}.pdb", f"{pdb_id}_{pep_chain}.pdb", "natmp.pdb"]
    for filename in intermediate_files:
        if os.path.isfile(filename):
            print(f"Deleting: {filename}")
            os.remove(filename)