from datasets import load_dataset

# 加载数据集
ds = load_dataset("yxchar/chemprot-tlm")

# 处理 label 列
def process_labels(example):
    # 将 0 转换为 false，其他值转换为 true
    example['label'] = 'false' if example['label'] == 0 else 'true'
    return example

# 应用处理函数
ds = ds.map(process_labels)

# 保存到 txt 文件，每行的格式为 text\tlabel
with open("chemprot_tlm_processed_test.txt", "w", encoding="utf-8") as f:
    for example in ds['test']:  # 如果数据集分为train/test/validation，请根据实际情况选择
        f.write(f"{example['text']}\t{example['label']}\n")
