import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# 读取训练数据
with open('./final_train.txt', 'r') as file:
    train_data = file.readlines()

train_texts = []
train_labels = []

for line in train_data:
    if line.strip():
        try:
            word, label = line.strip().split("\t")
            if len(word) == 1 and not word.isalnum():
                train_texts.append(word)
                train_labels.append("O")
            else:
                train_texts.append(word)
                train_labels.append(label)
        except ValueError:
            # Skip the line if it does not contain exactly two elements
            continue

# 初始化设备和模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/llama3")
base_model = AutoModelForCausalLM.from_pretrained("/data/aim_nuist/aim_zhujj/llama3")

tokenizer.pad_token = tokenizer.eos_token
config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
    encoder_dropout=0.1,
    encoder_num_layers=8,
    encoder_hidden_size=2048
)
model = get_peft_model(base_model, config)

# 构建微调文本
train_texts = ['''Generate BIO tags for each word in the given paragraph,. The BIO format uses the following labels:
•	B: Beginning of an entity
•	I: Inside of an entity
•	O: Outside of an entity
Please extract all chemicals, genes, and diseases mentioned in the paragraph. Provide the output in the format <word> - <BIO tag>, where each word is followed by its corresponding BIO tag.
''' + text for text in train_texts]

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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.to(device)

# 开始微调
epochs = 1
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()  # 记录开始时间

    # 使用 tqdm 创建训练循环的进度条
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
        for batch in tepoch:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            max_seq_length = input_ids.shape[1]

            # Pad the labels to match the input sequence length
            padded_labels = torch.nn.functional.pad(labels, (0, max_seq_length - labels.shape[1]), value=-100).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=padded_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Update the progress bar description with the current loss
            tepoch.set_postfix(loss=loss.item())

    train_losses.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {epoch_loss / len(train_loader):.4f}\n")

# 绘制损失曲线
plt.plot(np.arange(1, epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("training_loss_curve.png")

# 保存模型参数
torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/llama_bc2_pt_model.pt")

# 导入模型参数
base_model = AutoModelForCausalLM.from_pretrained("/data/aim_nuist/aim_zhujj/llama3")
model = get_peft_model(base_model, config)
model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/llama_bc2_pt_model.pt"))
model.to(device)

# 读取测试数据
with open("./final_test.txt", "r") as file:
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
with open("llama3_bc2_pt.txt", "w") as output_file:
    start_time = time.time()  # 记录开始时间

    # 使用 tqdm 创建测试循环的进度条
    with tqdm(test_data, desc="Testing", unit="line") as ttest:
        for i, line in enumerate(ttest):
            if line.strip():  # 跳过空行
                # 进行单词或标点的标签预测
                input_text = '''Generate BIO tags for each word in the given paragraph,. The BIO format uses the following labels:
•	B: Beginning of an entity
•	I: Inside of an entity
•	O: Outside of an entity
Please extract all chemicals, genes, and diseases mentioned in the paragraph. Provide the output in the format <word> - <BIO tag>, where each word is followed by its corresponding BIO tag.
''' + line.strip()
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                outputs = model.generate(input_ids)
                predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 写入预测结果到output.txt
                output_file.write(f"{line.strip()}\t{predicted_label}\n")
