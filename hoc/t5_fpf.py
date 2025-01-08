import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# 加载文本和标签
text_folder = './text'
labels_folder = './labels'

texts = []
labels = []

for filename in os.listdir(text_folder):
    if filename.endswith('.txt'):
        with open(os.path.join(text_folder, filename), 'r') as file:
            texts.append(file.read().strip())

        label_file = os.path.join(labels_folder, filename)
        with open(label_file, 'r') as file:
            labels.append(file.read().strip())

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 初始化 tokenizer 和模型
tokenizer = T5Tokenizer.from_pretrained("/share/home/aim/aim_zhujj/ul2", use_fast=False)
model = T5ForConditionalGeneration.from_pretrained("/share/home/aim/aim_zhujj/ul2")

# 构建训练和验证数据
train_texts = ["classify document: " + text for text in train_texts]
val_texts = ["classify document: " + text for text in val_texts]

train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 训练循环
epochs = 200
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
plt.savefig("loss_curve.png")

# 保存模型参数
torch.save(model.state_dict(), "ul_model.pt")

# 加载模型参数用于推理
model = T5ForConditionalGeneration.from_pretrained("/share/home/aim/aim_zhujj/ul2")
model.load_state_dict(torch.load("ul_model.pt"))
model.to(device)

# 测试数据上的评估
test_texts = val_texts  # 使用验证数据作为测试数据进行演示
test_labels = val_labels

test_texts = ["classify document: " + text for text in test_texts]
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)

test_dataset = Dataset(test_encodings, val_labels_encodings)  # 演示中使用 val_labels_encodings
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


# 定义 Jaccard 相似度计算函数
def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


# 预测标签并输出结果
similarities = []

with open("output.txt", "w") as output_file:
    for i, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
        predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

        similarity = jaccard_similarity(predicted_label, test_labels[i])
        similarities.append(similarity)

        output_file.write(
            f"Document: {test_texts[i]}\nPredicted Label: {predicted_label}\nTrue Label: {test_labels[i]}\nSimilarity: {similarity:.4f}\n\n"
        )

average_similarity = np.mean(similarities)
print(f"Average Similarity: {average_similarity:.4f}")
