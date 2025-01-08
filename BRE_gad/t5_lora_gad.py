import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model, PeftModel
# 定义文件夹范围
folders = [str(i) for i in range(1, 11)]

# 初始化 tokenizer 和模型
tokenizer = T5Tokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")
base_model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(base_model, lora_config)
# 加载并处理训练数据
train_texts = []
train_labels = []
invalid_lines_count = 0

for folder in folders:
    train_file = os.path.join(folder, 'train.tsv')
    if os.path.exists(train_file):
        with open(train_file, 'r') as file:
            train_data = file.readlines()

        for line in train_data:
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    word, label = parts
                    if len(word) == 1 and not word.isalnum():
                        train_texts.append(word)
                        train_labels.append("O")
                    else:
                        train_texts.append(word)
                        train_labels.append(label)
                else:
                    invalid_lines_count += 1

# 输出不符合格式的行数
print(f"Number of invalid lines: {invalid_lines_count}")
print(train_texts)
# 构建微调任务
train_texts = ['''Please determine whether there is a relationship between the @GENE$ and @DISEASE$ parts of the sentencePlease determine whether there is a relationship between the 1 and 2 parts of the sentence : ''' + text for text in train_texts]

# 对标签进行编码
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=1024)
train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt", max_length=1024)

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
torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5/t5lora_gad_model.pt")

# 导入模型参数
base_model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")
lora_model = get_peft_model(base_model, lora_config)
lora_model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5/t5lora_gad_model.pt"))

# 将模型移动到合适的设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 预测并处理测试数据

test_file = 'test1.tsv'
with open(test_file, 'r') as file:
    test_data = file.readlines()

with open(f"t5lora_gad_result.txt", "w") as output_file:
    for line in test_data:
        if line.strip():  # 跳过空行
            input_text = '''Please determine whether there is a relationship between the @GENE$ and @DISEASE$ parts of the sentencePlease determine whether there is a relationship between the 1 and 2 parts of the sentence : ''' + line.strip()
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids=input_ids)
            predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_file.write(f"{line.strip()}\t{predicted_label}\n")

print("测试完成！结果已保存到相应的txt文件中。")
