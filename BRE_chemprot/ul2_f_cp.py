import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np

# 加载训练数据
with open('./chemprot_train.txt', 'r') as file:
    train_data = file.readlines()
train_texts = []
train_labels = []

for line in train_data:
    if line.strip():
        word, label = line.strip().split("\t")
        if len(word) == 1 and not word.isalnum():
            train_texts.append(word)
            train_labels.append("O")
        else:
            train_texts.append(word)
            train_labels.append(label)

# 初始化 tokenizer 和模型
tokenizer = T5Tokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/ul2", use_fast=False)
model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/ul2")

# 构建微调任务
train_texts = ['''Please consider whether there is a relationship between the two entities within << >> and [[ ]] in the text. If a relationship exists, label it as true; otherwise, label it as false : '''+ text for text in train_texts]

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

train_dataset = Dataset(train_encodings, train_labels_encodings)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

# 定义微调参数和优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 开始微调
epochs=5
# 训练模型
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
torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/ul2/ul_cp_model.pt")

# 导入模型参数
model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/ul2")
model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/ul2/ul_cp_model.pt"))

# 将模型移动到合适的设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 读取测试数据
with open("./chemprot_test.txt", "r") as file:
    test_data = file.readlines()

# 初始化 tokenizer
tokenizer = T5Tokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/ul2", use_fast=False)

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
with open("ul2_cpresult.txt", "w") as output_file:
    for line in test_data:
        if line.strip():  # 跳过空行
            # 进行单词或标点的标签预测
            input_text = '''Please consider whether there is a relationship between the two entities within << >> and [[ ]] in the text. If a relationship exists, label it as true; otherwise, label it as false : ''' + line.strip()
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids)
            predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 写入预测结果到output.txt
            output_file.write(f"{line.strip()}\t{predicted_label}\n")
