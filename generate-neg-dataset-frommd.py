import os
import sys
import multiprocessing as mp
import glob
import shutil

input_path = "/rv2/biodata/md_decoys/MEGADOCK_output"
target_path = "/rv2/biodata/neg-dataset-3200"
positive_path = '/rv2/biodata/md-decoys-hq'
md_lastindex = 1000
num_neg = 3 #how many decoys to be copied per directory, must be less than 100

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def processdir(sub_path):
    global num_neg
    if not os.path.isfile(os.path.join(input_path, sub_path, 'dockQ_results_sorted.csv')):
        return
    with open(os.path.join(input_path, sub_path, 'dockQ_results_sorted.csv')) as dqscorefile:
        lines = dqscorefile.readlines()
        #process positives
        for line in lines:
            line = line.rstrip()
            fields = line.split(',')
            if (len(fields) >= 3):
                try:
                    score = float(fields[2])
                except ValueError:
                    continue
                if (score < 0.8): #sorted so stop when low number appears
                    break 
                fn = fields[0].split('/')[2]
                if not os.path.isfile(os.path.join(positive_path, fn)):
                    shutil.copy(os.path.join(input_path, sub_path, fn), positive_path)
                    #print(f'echo \'{fn},{score}\' >> ' + os.path.join(positive_path, 'dockq_scores.tsv'))
                    os.system(f'echo \'{fn},{score}\' >> ' + os.path.join(positive_path, 'dockq_scores.tsv'))
        #process negatives
        for index in range(-1,-100,-1):
            line = lines[index].rstrip()
            fields = line.split(',')
            if (len(fields) >= 3):
                try:
                    score = float(fields[2])
                except ValueError:
                    continue
                if (score > 0.3): #sorted so stop when big num appears 
                    break
                fn = fields[0].split('/')[2]
                if not os.path.isfile(os.path.join(target_path, fn)):
                    shutil.copy(os.path.join(input_path, sub_path, fn), target_path)
                    os.system(f'echo \'{fn},{score}\' >> ' + os.path.join(target_path, 'dockq_scores.tsv'))
                num_neg -= 1
                if (num_neg <= 0):
                    break


#    for file in glob.glob(os.path.join(input_path, sub_path, '*.pdb')):
#        if file.endswith(tuple(f"{i}.pdb" for i in range(md_lastindex+1-numperdir, md_lastindex+1))):
#            if not os.path.isfile(os.path.join(target_path, file)):
#                shutil.copy(file, target_path)


def generate_neg_dataset_frommd(src_path, num_procs = 10): 
    dirs = [x for x in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, x))]
    dirs.sort()
    total_dirs = len(dirs)  # 전체 파일 개수
    print(f'{total_dirs} sub directories to be processed.')
    for index in range(0, total_dirs, num_procs):
        lastindex = index+num_procs if (index + num_procs < total_dirs) else total_dirs 
        items = dirs[index:lastindex]

        print(f"Processing {index + 1}/{total_dirs}: {items}")   
        pool = mp.Pool(lastindex - index) 
        pool.map(processdir, items)
        pool.close()
        pool.join()
        if (num_procs != lastindex - index):
            print("last items:", items)

    return


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        input_path = sys.argv[1]
    if (len(sys.argv) > 2):
        target_path = sys.argv[2]
    num_neg = 3
    print(f"Assemblying dataset in {target_path} copied from {input_path}")
    generate_neg_dataset_frommd(input_path, 10)