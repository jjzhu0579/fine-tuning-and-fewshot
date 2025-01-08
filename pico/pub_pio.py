import os
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
# 文件夹路径
# doc_dir = './documents'
# aggregated_base = './aggregated/starting_spans'
# pio_types = ['interventions', 'outcomes', 'participants']
# subsets = ['train']
output_file_path = './duan1.txt'

# 合并所有文件内容并写入输出文件
# with open(output_file_path, 'w', encoding='utf-8') as output_file:
#     tokens_files = glob(os.path.join(doc_dir, '*.tokens'))
#     for tokens_file in tokens_files:
#         pmid = os.path.basename(tokens_file).split('.')[0]
#         doc_path = os.path.join(doc_dir, f'{pmid}.txt')
#         if not os.path.isfile(doc_path):
#             continue
#
#         with open(tokens_file, 'r', encoding='utf-8') as tf:
#             tokens_lines = tf.readlines()
#
#         aggregated_lines_dict = {}
#         for pio in pio_types:
#             aggregated_file = os.path.join(aggregated_base, pio, subsets[0], f'{pmid}.AGGREGATED.ann')
#             if os.path.isfile(aggregated_file):
#                 with open(aggregated_file, 'r', encoding='utf-8') as af:
#                     aggregated_lines = af.readlines()
#                     aggregated_lines_dict[pio] = aggregated_lines
#
#         if not all(len(tokens_lines) == len(aggregated_lines_dict[pio]) for pio in aggregated_lines_dict):
#             print(f"Warning: Number of lines do not match for {tokens_file} and .AGGREGATED.ann files")
#             continue
#
#         for idx in range(len(tokens_lines)):
#             combined_line = tokens_lines[idx].strip()
#             for pio in pio_types:
#                 if pio in aggregated_lines_dict:
#                     combined_line += ' ' + aggregated_lines_dict[pio][idx].strip()
#             output_file.write(f"{combined_line}\n")
#
# print("Merging completed.")
#
#
with open(output_file_path, 'r', encoding='utf-8') as file:
    data = file.readlines()

texts = []
labels = []

for line in data:
    if line.strip():
        parts = line.strip().split()
        if len(parts) > 1:
            texts.append(" ".join(parts[:-3]))  # 取单词部分
            labels.append(parts[-3:])  # 取标签部分

# 将数据划分为训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

# 初始化 tokenizer 和模型
model_name = '/data/aim_nuist/aim_zhujj/xinjian/PubMedBERT'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda')

# 构建微调任务
train_texts = ["generate IOB2 labels: " + text for text in train_texts]
train_labels = [" ".join(label) for label in train_labels]

# 对标签进行编码
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt", max_length=128)

# 定义数据集类
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels_encodings):
        self.encodings = encodings
        self.labels_encodings = labels_encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels_encodings['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)
# 定义数据集类
class CustomTestDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx].strip()
        encoding = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return encoding


train_dataset = Dataset(train_encodings, train_labels_encodings)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# 定义微调参数和优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 开始微调
epochs = 2
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        max_seq_length = labels.shape[1]
        input_ids = input_ids[:, :max_seq_length]
        attention_mask = attention_mask[:, :max_seq_length]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

# 绘制损失曲线
plt.plot(np.arange(1, epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("training_loss_curve.png")

# 保存模型参数
torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/PubMedBERT/pub_model.pt")

model_name = '/data/aim_nuist/aim_zhujj/xinjian/PubMedBERT'
model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda')
model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/PubMedBERT/pub_model.pt"))

# 将模型移动到合适的设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

with open("./test1.txt", "r") as file:
    test_data = file.readlines()

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 定义数据集类
class CustomTestDataset(Dataset):
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
with open("output_pub11.txt", "w") as output_file:
    for line in test_data:
        if line.strip():  # 跳过空行
            # 进行单词或标点的标签预测
            if line.strip():  # Skip empty lines
                input_text = "generate IOB2 label: " + line.strip()  # Generate input for the model with each line
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                tokenized = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(device)
                output = model(**tokenized, return_dict=True)
                predicted_labels = output.logits[0].argmax(dim=-1)  # Get the indices of the highest probabilities
                predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_labels)  # Convert IDs to tokens

            # 写入预测结果到output.txt
            output_file.write(f"{line.strip()} {predicted_tokens[0:4]}\n")

from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np

# 读取文件内容
def read_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    labels = [line.strip() for line in lines if line.strip()]
    return labels

# 计算 F1 分数
def compute_f1(predictions, true_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    return precision, recall, f1

# 文件路径
pred_file = './output1.txt'
true_file = 'test2.txt'

# 读取预测值和真实标签
predictions = read_labels(pred_file)
true_labels = read_labels(true_file)

# 检查预测值和真实标签长度是否一致
if len(predictions) != len(true_labels):
    raise ValueError("预测值和真实标签的行数不一致！")

# 计算 F1 分数
precision, recall, f1 = compute_f1(predictions, true_labels)

# 输出结果
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")