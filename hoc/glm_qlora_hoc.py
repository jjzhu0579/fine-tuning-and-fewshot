import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

device = "cuda"

# 加载文本和标签
text_folder = './text'
labels_folder = './labels'

texts = []
labels = []

for filename in os.listdir(text_folder):
    if filename.endswith('.txt'):
        with open(os.path.join(text_folder, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read().strip())

        label_file = os.path.join(labels_folder, filename)
        with open(label_file, 'r', encoding='utf-8') as file:
            labels.append(file.read().strip())

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 初始化 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained('/share/home/aim/aim_zhujj/glm4_lora_q8/ZhipuAI/glm-4-9b-chat', use_fast=False,
                                          trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained('/share/home/aim/aim_zhujj/glm4_lora_q8/ZhipuAI/glm-4-9b-chat',
                                             torch_dtype=torch.float32, load_in_8bit=True, device_map="sequential",
                                             trust_remote_code=True)
base_model.enable_input_require_grads()  # 开启梯度检查点

# 配置和加载LoRA模型
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # 现存问题只微调部分演示即可
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)  # 调整批次大小
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False)

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # 调整学习率
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 训练模型
epochs = 1  # 增加训练轮数
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
plt.savefig("t_lora_loss_curve.png")

# 保存模型参数
torch.save(model.state_dict(), "glm_qlora_model.pt")

# 加载模型参数用于推理
base_model = AutoModelForCausalLM.from_pretrained('/share/home/aim/aim_zhujj/glm4_lora_q8/ZhipuAI/glm-4-9b-chat',
                                             torch_dtype=torch.float32, load_in_8bit=True, device_map="sequential",
                                             trust_remote_code=True)
lora_model = get_peft_model(base_model, lora_config)
lora_model.load_state_dict(torch.load("glm_qlora_model.pt"))
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

# 过滤非ASCII字符的函数
def filter_non_ascii(text):
    return ''.join([char if ord(char) < 128 else '' for char in text])

# 预测标签并输出结果
f1_scores = []

with open("t_lora.txt", "w", encoding="utf-8") as output_file:  # 指定编码为utf-8
    for i, batch in enumerate(test_loader):
        # 获取输入
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 准备推理输入
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # 推理
        with torch.no_grad():
            outputs = lora_model.generate(**inputs, max_new_tokens=50)  # 使用max_new_tokens
            predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_label = filter_non_ascii(predicted_label)  # 过滤非ASCII字符

        # 计算 F1 分数
        f1 = compute_f1_score(predicted_label, test_labels[i])
        f1_scores.append(f1)

        # 写入文件
        output_file.write(
            f"Predicted Label: {predicted_label}\nTrue Label: {test_labels[i]}\nF1 Score: {f1:.4f}\n\n"
        )

average_f1_score = np.mean(f1_scores)
print(f"Average F1 Score: {average_f1_score:.4f}")
