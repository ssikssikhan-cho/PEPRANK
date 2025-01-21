import os
import shutil

# 경로 설정
megadock_output_dir = '/rv2/biodata/md_decoys/MEGADOCK_output'
pep_check_decoys_dir = '/rv2/biodata/peppro_decoys2'
output_base_dir = '/rv2/biodata/pep_dataset'
peppro_list_file = '/rv2/biodata/peppro_exsitunbound.list'


# 파일 복사 및 이름 변경 함수 (MEGADOCK과 pep 경로 처리 분리)
def copy_and_rename(src, dest, start_idx, end_idx, new_filename_template, is_megadock=False):
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    file_copied = False
    for i in range(start_idx, end_idx + 1):
        if is_megadock:
            # MEGADOCK의 경우 .out.decoy 형식
            src_file = os.path.join(src, f'{new_filename_template}.out.decoy.{i}.pdb')
            dest_file = os.path.join(dest, f'{new_filename_template}_model_{i}.pdb')
        else:
            # pep_check_decoys는 model 디렉토리 포함
            src_file = os.path.join(src, 'model', f'model_{i}.pdb')
            dest_file = os.path.join(dest, f'{new_filename_template}_model_{i+1000}.pdb')  # 1001~1010으로 이름 지정

        # 디버깅 메시지 추가
        if os.path.exists(src_file):
            try:
                shutil.copy(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}")
                file_copied = True
            except Exception as e:
                print(f"Failed to copy {src_file} to {dest_file}. Error: {e}")
        else:
            print(f"Source file not found: {src_file}")

    return file_copied

def checkexisting(src, dest, start_idx, end_idx, new_filename_template, is_megadock=False):
    #if not os.path.exists(dest):
    #    os.makedirs(dest)

    file_copied = True # folder not to be deleted
    for i in range(start_idx, end_idx + 1):
        if is_megadock:
            src_file = os.path.join(src, f'{new_filename_template}.out.decoy.{i}.pdb')
            dest_file = os.path.join(dest, f'{new_filename_template}_model_{i}.pdb')
        else:
            src_file = os.path.join(src, 'model', f'model_{i}.pdb')
            dest_file = os.path.join(dest, f'{new_filename_template}_model_{i+1000}.pdb')
        if os.path.isfile(dest_file):
            print(f'{dest_file} exists.')
            break
        #else:
        #    print(f"{dest_file} not found.")

    return file_copied

def copydockqscorefile_md(src, dest, new_filename_template):
    dockqscorefn = 'dockQ_results_sorted.csv'
    src_file = os.path.join(src, dockqscorefn)
    dest_file = os.path.join(dest, dockqscorefn)
    pdbid = new_filename_template.split('_')[0]
    
    if os.path.exists(src_file):
        try:
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")
            cmd = f'sed -i \'s/MEGADOCK_output\/{new_filename_template}\/{new_filename_template}\.out\.decoy\./{new_filename_template}_model_/g\' {dest_file}'
            os.system(cmd)
            cmd = f'sed -i \'s/,MEGADOCK_input\/original_PDB\/{pdbid}\.pdb//g\' {dest_file}'
            os.system(cmd)
        except Exception as e:
            print(f"Failed to copy {src_file} to {dest_file}. Error: {e}")
    else:
        print(f"Source file not found: {src_file}")

def append_dockqscorefile_gpd(src, dest, new_filename_template):
    dockqscorefn = 'dockQ_results_sorted.csv'
    dest_file = os.path.join(dest, dockqscorefn)
    pdbid = new_filename_template.split('_')[0]
    gpd_dockqscorefn = 'dockqscores.csv'
    src_file = os.path.join(src, 'model', gpd_dockqscorefn)

    if not os.path.isfile(dest_file):
        print(f'file not found: {dest_file}')
        return
    if not os.path.isfile(src_file):
        print(f'file not found: {src_file}')
        return

    i = 1
    with open(src_file, 'r') as file:
        for line in file:
            columns = line.strip().split(",")
            os.system('echo '+f'{new_filename_template}_model_{i+1000}.pdb,'+columns[1]+' >> '+dest_file)
            #print('echo '+f'{new_filename_template}_model_{i+1000}.pdb,'+columns[1]+' >> '+dest_file)
            i += 1

# peppro_exsitunbound.list 파일 처리
def process_peppro_list_file():
    with open(peppro_list_file, 'r') as file:
        for line in file:
            columns = line.strip().split("\t")
            if len(columns) >= 4:
                first_column = columns[0]
                second_column = columns[1]
                third_column = columns[2]
                fourth_column = columns[3]

                # pep_check_decoys 디렉토리 확인
                pep_base_name = second_column[:4]  # 두 번째 열에서 앞 네 글자 추출
                megadock_dir_name = f'{first_column[:4].upper()}_{first_column.split(":")[1][0]}_{third_column}_{first_column.split(":")[0][-1]}'
                list.append(megadock_dir_name)

                megadock_files_src = os.path.join(megadock_output_dir, megadock_dir_name)
                pep_check_dir = os.path.join(pep_check_decoys_dir, pep_base_name)
                
                if os.path.exists(pep_check_dir):
#                    print(f'{pep_base_name} found in pep_check_decoys.')

                    # 파일 복사 및 이름 변경 (pep_check_decoys의 model 경로 포함)
                    output_dir = os.path.join(output_base_dir, megadock_dir_name)
#                    file_copied = copy_and_rename(megadock_files_src, output_dir, 1, 1000, megadock_dir_name, is_megadock=True)
#                    file_copied |= copy_and_rename(pep_check_dir, output_dir, 1, 10, megadock_dir_name, is_megadock=False)
                    file_copied = checkexisting(pep_check_dir, output_dir, 1, 10, megadock_dir_name, is_megadock=False)
#                    copydockqscorefile_md(megadock_files_src, output_dir, megadock_dir_name)
#                    append_dockqscorefile_gpd(pep_check_dir, output_dir, megadock_dir_name)

                    # 복사된 파일이 없을 경우 빈 폴더 삭제
                    if not file_copied and os.path.exists(output_dir):
                        shutil.rmtree(output_dir)
                        print(f'빈 폴더 {output_dir}가 삭제되었습니다.')
                else:
                    print(f'{pep_base_name} does not exist in {pep_check_decoys_dir}.')


def find_copy_mostsimilars_peppro_list():
    os.chdir(megadock_output_dir)
    list = []
    with open(peppro_list_file, 'r') as file:
        for line in file:
            columns = line.strip().split("\t")
            if len(columns) >= 4:
                first_column = columns[0]
                second_column = columns[1]
                third_column = columns[2]
                fourth_column = columns[3]

                # pep_check_decoys 디렉토리 확인
                pep_base_name = second_column[:4]  # 두 번째 열에서 앞 네 글자 추출
                #megadock_dir_name = f'{first_column[:4].upper()}_{first_column.split(":")[1][0]}_{third_column}_{first_column.split(":")[0][-1]}'
                megadock_dir_name = f'{first_column[:4].upper()}_{first_column.split(":")[1][0]}'
                # list all sub directories starting with megadock_dir_name in megadock_output_dir
                subdirs = [d for d in os.listdir(megadock_output_dir) if d.startswith(megadock_dir_name) and os.path.isdir(os.path.join(megadock_output_dir, d))]
                list.extend(subdirs)

                for megadock_dir_name in subdirs:
                    megadock_files_src = os.path.join(megadock_output_dir, megadock_dir_name)
                    pep_dataset_dir = os.path.join(output_base_dir, megadock_dir_name)
                    
                    if not os.path.exists(pep_dataset_dir):
    #                    print(f'{pep_base_name} found in pep_check_decoys.')

                        # 파일 복사 및 이름 변경 (pep_check_decoys의 model 경로 포함)
    #                    output_dir = os.path.join(output_base_dir, megadock_dir_name)
                        file_copied = copy_and_rename(megadock_files_src, pep_dataset_dir, 1, 1000, megadock_dir_name, is_megadock=True)
    #                    file_copied |= copy_and_rename(pep_check_dir, output_dir, 1, 10, megadock_dir_name, is_megadock=False)
    #                    file_copied = checkexisting(pep_check_dir, output_dir, 1, 10, megadock_dir_name, is_megadock=False)
                        copydockqscorefile_md(megadock_files_src, pep_dataset_dir, megadock_dir_name)
    #                    append_dockqscorefile_gpd(pep_check_dir, output_dir, megadock_dir_name)

                        # 복사된 파일이 없을 경우 빈 폴더 삭제
                        if not file_copied and os.path.exists(output_dir):
                            shutil.rmtree(output_dir)
                            print(f'빈 폴더 {output_dir}가 삭제되었습니다.')
                    else:
                        print(f'{pep_dataset_dir} exists.')

                # multi-chain receptor case
                try:
                    megadock_dir_name = f'{first_column[:4].upper()}_{first_column.split(":")[1][1]}'
                except:
                    continue
                subdirs = [d for d in os.listdir(megadock_output_dir) if d.startswith(megadock_dir_name) and os.path.isdir(os.path.join(megadock_output_dir, d))]
                list.extend(subdirs)
                for megadock_dir_name in subdirs:
                    megadock_files_src = os.path.join(megadock_output_dir, megadock_dir_name)
                    pep_dataset_dir = os.path.join(output_base_dir, megadock_dir_name)                    
                    if not os.path.exists(pep_dataset_dir):
                        file_copied = copy_and_rename(megadock_files_src, pep_dataset_dir, 1, 1000, megadock_dir_name, is_megadock=True)
                        copydockqscorefile_md(megadock_files_src, pep_dataset_dir, megadock_dir_name)
                        if not file_copied and os.path.exists(pep_dataset_dir):
                            shutil.rmtree(pep_dataset_dir)
                            print(f'deleted empty folder {pep_dataset_dir}.')
                    else:
                        print(f'{pep_dataset_dir} exists.')

    print(list)

if __name__ == "__main__":
    #process_peppro_list_file()
    find_copy_mostsimilars_peppro_list()
