import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 加载数据集
def load_dataset(filepath):
    texts = []
    labels = []
    if os.path.isfile(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                question = item["questions"]
                context = item["context"]
                answer = item["answer"]
                print(answer)
                input_text = f"Please answer the following QUESTION: {question}  with yes or no according to the CONTEXT: {context}"

                texts.append(input_text)
                labels.append(answer)
    return texts, labels


# 加载训练数据
train_texts, train_labels = load_dataset('./Task7B_yesno_train.json')

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1,
                                                                    random_state=42)

# 初始化 tokenizer 和模型
tokenizer = T5Tokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")
model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")

# 构建训练和验证数据
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors = "pt", max_length = 512)
train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt", max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
val_labels_encodings = tokenizer(val_labels, truncation=True, padding=True, return_tensors="pt", max_length=512)


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


train_dataset = Dataset(train_encodings, train_labels_encodings)
val_dataset = Dataset(val_encodings, val_labels_encodings)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 训练循环
epochs = 20
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
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
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))

    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            max_seq_length = labels.shape[1]
            input_ids = input_ids[:, :max_seq_length]
            attention_mask = attention_mask[:, :max_seq_length]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_val_loss += loss.item()
    val_losses.append(epoch_val_loss / len(val_loader))

# 绘制损失曲线
plt.plot(np.arange(1, epochs + 1), train_losses, label="Training Loss")
plt.plot(np.arange(1, epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.savefig("t5_loss_curve.png")

# 保存模型参数
torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5/t5_bqa_model.pt")

# 加载模型参数用于推理
model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")
model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5/t5_bqa_model.pt"))
model.to(device)

# 读取测试数据
test_texts, test_labels = load_dataset('Task7B_yesno_test.json')

test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


test_dataset = TestDataset(test_encodings)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Predict labels and save results
results = []
with open("t.txt", "w", encoding='utf-8') as output_file:
    for i, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
        predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

        output_file.write(f'"{predicted_label}",\n')

