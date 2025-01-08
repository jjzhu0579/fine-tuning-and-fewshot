import os
from glob import glob

# 文件夹路径
doc_dir = './documents'
aggregated_base = './aggregated/starting_spans'

# PIO 类别
pio_types = ['interventions', 'outcomes', 'participants']
subsets = ['train']

# 合并后的文件路径
output_file_path = './merged_output.txt'

# 打开输出文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:

    # 获取所有 .tokens 文件路径
    tokens_files = glob(os.path.join(doc_dir, '*.tokens'))

    # 遍历每个 .tokens 文件
    for tokens_file in tokens_files:
        pmid = os.path.basename(tokens_file).split('.')[0]
        doc_path = os.path.join(doc_dir, f'{pmid}.txt')

        # 如果对应的 .txt 文件不存在，跳过该文件
        if not os.path.isfile(doc_path):
            continue

        # 读取 .tokens 文件内容
        with open(tokens_file, 'r', encoding='utf-8') as tf:
            tokens_lines = tf.readlines()

        # 读取 .AGGREGATED.ann 文件内容
        aggregated_lines_dict = {}
        for pio in pio_types:
            aggregated_file = os.path.join(aggregated_base, pio, subsets[0], f'{pmid}.AGGREGATED.ann')
            if not os.path.isfile(aggregated_file):
                continue
            with open(aggregated_file, 'r', encoding='utf-8') as af:
                aggregated_lines = af.readlines()
                aggregated_lines_dict[pio] = aggregated_lines

        # 检查 .tokens 文件和 .AGGREGATED.ann 文件的行数是否相等
        if not all(len(tokens_lines) == len(aggregated_lines_dict[pio]) for pio in aggregated_lines_dict):
            print(f"Warning: Number of lines do not match for {tokens_file} and .AGGREGATED.ann files")
            continue

        # 合并每一行并写入输出文件
        for idx in range(len(tokens_lines)):
            combined_line = tokens_lines[idx].strip()
            for pio in pio_types:
                if pio in aggregated_lines_dict:
                    combined_line += ' ' + aggregated_lines_dict[pio][idx].strip()
            output_file.write(f"{combined_line}\n")

print("Merging completed.")
