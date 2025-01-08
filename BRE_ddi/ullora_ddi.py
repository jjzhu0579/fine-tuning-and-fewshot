import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
def read_data_from_txt(file_path):
    sentences = []
    labels = []
    sentence_ids = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                print(f"Ignoring line: {line}")
                continue
            sentence_id = parts[0]
            text = " ".join(parts[1:-1])
            label = parts[-1]
            sentences.append(f"extract relationship: {text}")
            labels.append(label)
            sentence_ids.append(sentence_id)
    return sentences, labels, sentence_ids

# 加载训练数据
train_sentences, train_labels, _ = read_data_from_txt('output_train2.txt')

# 检查一些数据，以确保加载和格式化正确
print(f"Loaded {len(train_sentences)} training sentences.")

# 初始化 tokenizer 和模型
tokenizer = T5Tokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/ul2")
base_model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/ul2")
# 设置 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]
)

# 获取 LoRA 微调模型
model = get_peft_model(base_model, lora_config)

# 对数据进行编码
if len(train_sentences) > 0:
    train_encodings = tokenizer(train_sentences, truncation=True, padding=True, return_tensors="pt", max_length=128)
    train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt", max_length=128)
else:
    raise ValueError("No valid training data loaded.")

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
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# # 开始微调
# epochs = 20
# train_losses = []
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0
#     for batch in train_loader:
#         optimizer.zero_grad()
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     train_losses.append(epoch_loss / len(train_loader))
#
# # 绘制损失曲线
# plt.plot(np.arange(1, epochs + 1), train_losses, label="Training Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training Loss Curve")
# plt.legend()
# plt.savefig("training_loss_curve.png")
#
# # 保存模型参数
# torch.save(model.state_dict(), "ul_lora_model.pt")

# 导入模型参数
base_model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/ul2")
model = get_peft_model(base_model, lora_config)
model.load_state_dict(torch.load("ul_lora_model.pt"))
model.to(device)

# 读取测试数据
test_sentences, test_labels, test_sentence_ids = read_data_from_txt('output_test2.txt')
val_labels = test_labels
# 定义数据集类
class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return encoding

# 创建数据加载器
test_dataset = CustomTestDataset(test_sentences, tokenizer)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
# 初始化 MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import re
mlb = MultiLabelBinarizer()
def extract_true_false(text):
    match = re.search(r'true|false', text)
    if match:
        return match.group()
    else:
        return None

# 训练 MultiLabelBinarizer
all_labels = train_labels + val_labels + test_labels
mlb.fit([label.split() for label in all_labels])


# 预测标签并输出到 output.txt，并计算 F1 分数
predicted_labels = []
f1_scores = []
i = 0
# 计算 F1 分数的函数，加入空格删除功能
def compute_f1_score(pred_label, true_label):
    # 删除预测标签和真实标签中的字母间空格
    pred_label = "".join(pred_label.split())
    true_label = "".join(true_label.split())
    pred_label = extract_true_false(pred_label)
    true_label = extract_true_false(true_label)
    print(i,pred_label)
    print(i,true_label)
    if pred_label is None or true_label is None:
        return 0.0  # 或者可以根据需要返回其他值
    pred_tokens = pred_label.split()
    true_tokens = true_label.split()
    pred_binarized = mlb.transform([pred_tokens])
    true_binarized = mlb.transform([true_tokens])
    return f1_score(true_binarized[0], pred_binarized[0], average='macro')


with open("output_t.txt", "w") as output_file:
    for sentence_id, sentence in zip(test_sentence_ids, test_sentences):
        input_text = sentence
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids=input_ids, max_length=128)
        predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 删除预测标签中的字母间空格
        predicted_label = "".join(predicted_label.split())

        # 计算 F1 分数
        true_label = test_labels[test_sentence_ids.index(sentence_id)]
        f1 = compute_f1_score(predicted_label, true_label)
        f1_scores.append(f1)

        # 去掉生成的文本前缀 "extract relationship: {text}"
        predicted_label = predicted_label.replace("extract relationship:", "").strip()
        predicted_labels.append(predicted_label)

        # 将预测结果写入 output.txt
        output_file.write(f"{sentence_id}\t{predicted_label}\n")
# 计算平均 F1 分数
average_f1_score = np.mean(f1_scores)
print(f"Average F1 Score: {average_f1_score:.4f}")

# 输出预测结果的 F1 评估信息到 t_fpf.txt
with open("t_fpf.txt", "w") as f1_output_file:
    for i in range(len(predicted_labels)):
        f1_output_file.write(
            f"Predicted Label: {predicted_labels[i]}\nTrue Label: {test_labels[i]}\nF1 Score: {f1_scores[i]:.4f}\n\n"
        )