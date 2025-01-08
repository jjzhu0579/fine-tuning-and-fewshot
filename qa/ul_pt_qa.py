import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType
import os

# Step 1: 读取 pqal_fold0 到 pqal_fold9 下的所有 train_set.txt 和 dev_set.txt 数据
train_data = []

for i in range(10):
    folder_name = f'./pqal_fold{i}'
    for file_name in ['train_set.txt', 'dev_set.txt']:
        file_path = os.path.join(folder_name, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                train_data.extend(file.readlines())

# Step 2: 数据处理
train_texts = []
train_labels = []
invalid_lines_count = 0

for line in train_data:
    if line.strip():
        parts = line.strip().split("\t")
        # 检查是否有两个部分
        if len(parts) == 2:
            word, label = parts
            if len(word) == 1 and not word.isalnum():
                train_texts.append(word)
                train_labels.append("O")
            else:
                train_texts.append(word)
                train_labels.append(label)
        else:
            # 记录不符合格式的行
            invalid_lines_count += 1

# Step 3: 加载模型和 tokenizer
tokenizer = T5Tokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/ul2", use_fast=False, trust_remote_code=True)
model = T5ForConditionalGeneration.from_pretrained(
    "/data/aim_nuist/aim_zhujj/xinjian/ul2",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# 配置 PromptEncoder
config = PromptEncoderConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=20,
    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
    encoder_dropout=0.1,
    encoder_num_layers=8,
    encoder_hidden_size=2048
)
model = get_peft_model(model, config)

# Step 4: 构建微调任务
train_texts = ['''Please answer the following QUESTION with yes or no or maybe according to the CONTEXT:''' + text for text in train_texts]

# 对标签进行编码
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt", max_length=128)

# Step 5: 定义数据集类
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

train_dataset = Dataset(train_encodings, train_labels_encodings)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Step 6: 定义微调参数和优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Step 7: 开始微调
epochs = 5
train_losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
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
torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/ul2/ul2_qa_pt_model.pt")

# Step 8: 导入模型参数
model = T5ForConditionalGeneration.from_pretrained(
    "/data/aim_nuist/aim_zhujj/xinjian/ul2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()
model = get_peft_model(model, config)
model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/ul2/ul2_qa_pt_model.pt"))

# 将模型移动到合适的设备
model.to(device)

# Step 9: 读取测试数据
with open("./test_set.txt", "r") as file:
    test_data = file.readlines()

# 定义测试数据集类
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

# Step 10: 预测标签并输出到output.txt
with open("ul2_pt_qa.txt", "w") as output_file:
    for line in test_data:
        if line.strip():  # 跳过空行
            # 进行单词或标点的标签预测
            input_text = '''Please answer the following QUESTION with yes or no or maybe according to the CONTEXT:''' + line.strip()
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids=input_ids)
            predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 写入预测结果到output.txt
            output_file.write(f"{line.strip()}\t{predicted_label}\n")
