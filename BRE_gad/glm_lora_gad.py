import os
import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model, PeftModel
# 定义文件夹范围
folders = [str(i) for i in range(1, 11)]

# 初始化 tokenizer 和模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 初始化 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained('/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat',
                                          trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()
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
train_texts = ['''Please determine whether there is a relationship between the 1 and 2 parts of the sentence : ''' + text for text in train_texts]

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

# # 开始微调
# epochs = 3
# train_losses = []
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0
#     for batch in train_loader:
#         optimizer.zero_grad()
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)
#         max_seq_length = input_ids.shape[1]
#
#         # Pad the labels to match the input sequence length
#         padded_labels = torch.nn.functional.pad(labels, (0, max_seq_length - labels.shape[1]), value=-100).to(device)
#
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=padded_labels)
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
# torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat/glmlora_gad_model.pt")

# 导入模型参数
base_model = AutoModelForCausalLM.from_pretrained(
    '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

lora_model = get_peft_model(base_model, lora_config)
lora_model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat/glmlora_gad_model.pt"))

# 将模型移动到合适的设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 预测并处理测试数据

from tqdm import tqdm
import torch

test_file = 'test1.tsv'

# 读取测试数据
try:
    with open(test_file, 'r') as file:
        test_data = file.readlines()
    print(f"成功读取测试文件，共 {len(test_data)} 行。")
except Exception as e:
    print(f"读取测试文件时出错：{e}")
    exit(1)  # 提前退出

# 检查设备是否正确配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备：{device}")

# 确保模型和分词器在正确的设备上
model.to(device)

# 打开输出文件，并使用进度条写入结果
try:
    with open(f"glmlora_gad_result.txt", "w", encoding="utf-8") as output_file:  # 指定 UTF-8 编码
        for line in tqdm(test_data, desc="Processing", unit="lines"):  # 在此处添加 tqdm
            if line.strip():  # 跳过空行
                input_text = '''Please determine whether there is a relationship between the 1 and 2 parts of the sentence : ''' + line.strip()

                # 调试输出
                print(f"正在处理行: {line.strip()}")  # 打印正在处理的行
                try:
                    # 对输入进行分词
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                    print(f"输入 ID 形状: {input_ids.shape}")  # 检查输入形状

                    # 使用模型生成输出
                    print("正在生成输出...")
                    outputs = model.generate(input_ids=input_ids, max_new_tokens=50)  # 添加 max_new_tokens 限制生成长度
                    predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"预测标签: {predicted_label}")  # 打印预测标签

                    # 写入文件并立即刷新缓冲区
                    output_file.write(f"{line.strip()}\t{predicted_label}\n")
                    output_file.flush()  # 刷新输出缓冲区

                except Exception as e:
                    print(f"处理行时出错: {line.strip()}\n错误: {e}")  # 如果出错则打印错误信息
                    continue  # 跳过该行并继续处理下一行

except Exception as e:
    print(f"写入文件时出错：{e}")

print("测试完成！结果已保存到相应的 txt 文件中。")
