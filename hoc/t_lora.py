import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import MultiLabelBinarizer

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
tokenizer = T5Tokenizer.from_pretrained("/share/home/aim/aim_zhujj/FLAN-T5")
base_model = T5ForConditionalGeneration.from_pretrained("/share/home/aim/aim_zhujj/FLAN-T5")

# 配置和加载LoRA模型
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,  # 增加Dropout
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(base_model, lora_config)

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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)  # 调整批次大小
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)  # 调整学习率
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 训练模型
epochs = 5  # 增加训练轮数
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
plt.savefig("t_lora_loss_curve.png")

# 保存模型参数
torch.save(model.state_dict(), "t_lora_model.pt")

# 加载模型参数用于推理
base_model = T5ForConditionalGeneration.from_pretrained("/share/home/aim/aim_zhujj/FLAN-T5")
lora_model = get_peft_model(base_model, lora_config)
lora_model.load_state_dict(torch.load("t_lora_model.pt"))
lora_model.to(device)

# 读取测试数据
test_texts = val_texts  # 使用验证数据作为测试数据进行演示
test_labels = val_labels

test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)

test_dataset = Dataset(test_encodings, val_labels_encodings)  # 演示中使用 val_labels_encodings
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化 MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# 将标签 binarize
all_labels = train_labels + val_labels + test_labels
mlb.fit([label.split() for label in all_labels])

# 计算 F1 分数
def compute_f1_score(pred_label, true_label):
    pred_tokens = pred_label.split()
    true_tokens = true_label.split()
    pred_binarized = mlb.transform([pred_tokens])
    true_binarized = mlb.transform([true_tokens])
    return f1_score(true_binarized[0], pred_binarized[0], average='macro')

# 预测标签并输出结果
f1_scores = []

with open("t_lora.txt", "w") as output_file:
    for i, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = lora_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
        predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

        f1 = compute_f1_score(predicted_label, test_labels[i])
        f1_scores.append(f1)

        output_file.write(
            f"Predicted Label: {predicted_label}\nTrue Label: {test_labels[i]}\nF1 Score: {f1:.4f}\n\n"
        )

average_f1_score = np.mean(f1_scores)
print(f"Average F1 Score: {average_f1_score:.4f}")
