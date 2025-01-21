import pandas as pd
import os

os.chdir("/workspace/DeepLT/Reranking/Pep_Pro")

# CSV 파일 경로 설정
#input_file_path = 'dockQ_results.csv' 원래는 이게 맞는데 중간에 파일 잃어버려서 어쩔 수 없이 아래 거 사용함
input_file_path = 'sorted_dockq_pdb.csv'

# CSV 파일 읽기
df = pd.read_csv(input_file_path)

# 전체 행 개수 출력
total_rows = len(df)
print(f"전체 행 개수: {total_rows}")

# 3번째 열이 비어있는 행 개수 출력
empty_third_col_count = df['best_dockQ'].isnull().sum()
print(f"3번째 열이 비어있는 행 개수: {empty_third_col_count}")

"""
# 1. DockQ_PDB 이름 순으로 정렬된 CSV 파일 저장
sorted_df = df.sort_values(by='DockQ_PDB')
sorted_df.to_csv('sorted_dockq_pdb.csv', index=False)
"""

# 2. 3번째 열의 값이 존재하는 애들로만 구성된 CSV 파일 저장
non_empty_third_col_df = df[df['best_dockQ'].notnull()]
non_empty_third_col_df.to_csv('non_empty_dockq.csv', index=False)

# 3. 2번 파일에서 3번째 열의 값이 0.49보다 작은 애들로만 구성된 CSV 파일 저장
filtered_df = non_empty_third_col_df[non_empty_third_col_df['best_dockQ'] < 0.49]
filtered_df.to_csv('0.49_dockq.csv', index=False)

print("파일 생성이 완료되었습니다.")
