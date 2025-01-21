import os

# 작업 디렉토리 설정
os.chdir("/workspace/DeepLT/Reranking/Pep_Pro")

# list 파일 열기
try:
    listfile = open('peppro_exsitunbound.list', 'r')
except IOError:
    print("list/output file IO error")
    quit()

# 다운로드 경로 설정
download_path = "/workspace/DeepLT/vol/Pep_Pro/peppro_exsitunbound_dataset"
os.makedirs(download_path, exist_ok=True)  # 경로가 없으면 생성

while True:
    # 한줄 읽기
    listline = listfile.readline()
    # 없으면 끝내기
    if not listline:
        break

    # 공백으로 나누기
    fd = listline.split()
    
    boundprotbase = fd[0].split('_')[0]  # 1dkd
    boundpepchain = fd[0].split('_')[1][0]  # E
    boundprotchain = fd[0].split(':')[1][0]  # A
    
    # 결과물은 다 체인 이름을 AB로 할 거임
    native = os.path.join(download_path, boundprotbase + "AB.pdb")  # 경로 수정

    # 1dkd라는 파일이 없으면 다운로드
    if not (os.path.isfile(os.path.join(download_path, boundprotbase + ".pdb"))):
        cmd = f"wget -P {download_path} https://files.rcsb.org/download/{boundprotbase}.pdb"  # 출력 제거
        print(cmd)
        os.system(cmd)

    cmd = f"pdb_selchain -{boundprotchain} {os.path.join(download_path, boundprotbase + '.pdb')} > " \
          f"{os.path.join(download_path, boundprotbase + boundprotchain + '.pdb')}"  # 경로 수정
    print(cmd)
    os.system(cmd)
    nAfile = os.path.join(download_path, boundprotbase + boundprotchain + ".pdb")  # 경로 수정
    
    # 체인 이름이 A가 아닌 경우 A로 변경
    if (boundprotchain != 'A'):
        cmd = f"pdb_rplchain -{boundprotchain}:A {nAfile} > " \
              f"{os.path.join(download_path, boundprotbase + boundprotchain + '_A.pdb')}"  # 경로 수정
        print(cmd)
        os.system(cmd)
        nAfile = os.path.join(download_path, boundprotbase + boundprotchain + '_A.pdb')  # 경로 수정
        
    cmd = f"pdb_selchain -{boundpepchain} {os.path.join(download_path, boundprotbase + '.pdb')} > " \
          f"{os.path.join(download_path, boundprotbase + boundpepchain + '.pdb')}"  # 경로 수정
    print(cmd)
    os.system(cmd)
    nBfile = os.path.join(download_path, boundprotbase + boundpepchain + ".pdb")  # 경로 수정
    
    # 체인 이름이 B가 아닌 경우 B로 변경
    if (boundpepchain != 'B'):
        cmd = f"pdb_rplchain -{boundpepchain}:B {nBfile} > " \
              f"{os.path.join(download_path, boundprotbase + boundpepchain + '_B.pdb')}"  # 경로 수정
        print(cmd)
        os.system(cmd)
        nBfile = os.path.join(download_path, boundprotbase + boundpepchain + '_B.pdb')  # 경로 수정

    # 두 PDB 합치기
    cmd = f"pdb_merge {nAfile} {nBfile} > {os.path.join(download_path, 'natmp.pdb')}"  # 경로 수정
    print(cmd)
    os.system(cmd)
    
    cmd = f"pdb_tidy {os.path.join(download_path, 'natmp.pdb')} > {native}"  # 경로 수정
    print(cmd)
    os.system(cmd)

# AB가 아닌 PDB 파일 삭제
for filename in os.listdir(download_path):
    # 파일의 확장자가 .pdb인지 확인
    if filename.endswith(".pdb"):
        # "AB.pdb"로 끝나는지 확인
        if not filename.endswith("AB.pdb"):
            print(f"Deleting: {filename}")
            os.remove(os.path.join(download_path, filename))  # 경로 수정
