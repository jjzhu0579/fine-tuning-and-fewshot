import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 初始化 tokenizer 和模型
tokenizer = T5Tokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")
model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")

# 读取测试数据
with open("./test.tsv.txt", "r") as file:
    test_data = file.readlines()

# 定义数据集类
class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx].strip()
        encoding = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return encoding

# 创建数据加载器
test_dataset = CustomTestDataset(test_data, tokenizer)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

# 预测标签并输出到output.txt
with open("t5_bc5chem_result.txt", "w") as output_file:
    for line in test_data:
        if line.strip():  # 跳过空行
            # 构建zero-shot输入文本
            input_text = '''Generate BIO tags for each word in the given paragraph,. The BIO format uses the following labels:
• B-<Entity>: Beginning of an entity
• I-<Entity>: Inside of an entity
• O: Outside of an entity
Please extract all chemicals, genes, and diseases mentioned in the paragraph. Provide the output in the format <word> - <BIO tag>, where each word is followed by its corresponding BIO tag.
''' + line.strip()

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids)
            predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 写入预测结果到output.txt
            output_file.write(f"{line.strip()}\t{predicted_label}\n")

print("Zero-shot inference completed and results are saved to 't5_bc5chem_result.txt'.")
