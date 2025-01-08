import os
import pandas as pd

# 文件夹范围
folders = [str(i) for i in range(1, 11)]

# 遍历每个文件夹
for folder in folders:
    for file_name in os.listdir(folder):
        # 只处理 .tsv 和 .csv 文件
        if file_name.endswith(".tsv") or file_name.endswith(".csv"):
            file_path = os.path.join(folder, file_name)

            # 读取文件，跳过第一行
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()[1:]  # 跳过第一行

            # 处理每一行，删除第一个 \t 及其之前的字符，并替换 1 和 0
            processed_lines = []
            for line in lines:
                # 找到第一个 \t 的位置，并截取 \t 后的部分
                line = line.split('\t', 1)[1]
                # 替换 \t 后的 1 为 true，0 为 false
                line = line.replace('\t1', '\ttrue').replace('\t0', '\tfalse')
                processed_lines.append(line)

            # 将处理后的内容写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(processed_lines)

print("文件处理完毕！")
