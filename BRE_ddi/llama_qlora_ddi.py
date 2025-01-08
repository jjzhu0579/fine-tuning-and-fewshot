import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType,LoraConfig
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# 读取训练数据
with open('./combined_train.txt', 'r') as file:
    train_data = file.readlines()

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

# 输出不符合格式的行数
print(f"Number of invalid lines: {invalid_lines_count}")
# 初始化设备和模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/llama3")
base_model = AutoModelForCausalLM.from_pretrained("/data/aim_nuist/aim_zhujj/llama3")

tokenizer.pad_token = tokenizer.eos_token
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
model = get_peft_model(base_model, lora_config)

# 构建微调文本
train_texts = ['''Ask whether there is a relationship between the two entities of the sentence: ''' + text for text in train_texts]

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# 定义微调参数和优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.to(device)

# # 开始微调
# epochs = 2
# train_losses = []
#
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0
#     start_time = time.time()  # 记录开始时间
#
#     # 使用 tqdm 创建训练循环的进度条
#     with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
#         for batch in tepoch:
#             optimizer.zero_grad()
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
#             max_seq_length = input_ids.shape[1]
#
#             # Pad the labels to match the input sequence length
#             padded_labels = torch.nn.functional.pad(labels, (0, max_seq_length - labels.shape[1]), value=-100).to(device)
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=padded_labels)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#
#             # Update the progress bar description with the current loss
#             tepoch.set_postfix(loss=loss.item())
#
#     train_losses.append(epoch_loss / len(train_loader))
#     print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {epoch_loss / len(train_loader):.4f}\n")
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
# torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/llama_ddi_qlora_model.pt")

# 导入模型参数
base_model = AutoModelForCausalLM.from_pretrained("/data/aim_nuist/aim_zhujj/llama3")
model = get_peft_model(base_model, lora_config)
model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/llama_ddi_qlora_model.pt"))
model.to(device)

# 读取测试数据
with open("./100.txt", "r") as file:
    test_data = file.readlines()

# 定义测试数据集类
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
with open("llama_ddi_qlora.txt", "w") as output_file:
    start_time = time.time()  # 记录开始时间

    # 使用 tqdm 创建测试循环的进度条
    with tqdm(test_data, desc="Testing", unit="line") as ttest:
        for i, line in enumerate(ttest):
            if line.strip():  # 跳过空行
                # 进行单词或标点的标签预测
                input_text = '''Ask whether there is a relationship between the two entities of the sentence: ''' + line.strip()
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                outputs = model.generate(input_ids)
                predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 写入预测结果到output.txt
                output_file.write(f"{line.strip()}\t{predicted_label}\n")
