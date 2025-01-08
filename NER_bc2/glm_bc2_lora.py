import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from peft import LoraConfig, get_peft_model, PeftModel,TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

with open('./final_train.txt', 'r') as file:
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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 初始化 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained('/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat', trust_remote_code=True)
# base_model = AutoModelForCausalLM.from_pretrained(
#     '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat',
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# ).to(device).eval()
# # 配置和加载LoRA模型
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # 现存问题只微调部分演示即可
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)
# model = get_peft_model(base_model, lora_config)

# 构建微调任务
train_texts = ['''Generate BIO tags for each word in the given paragraph,. The BIO format uses the following labels:
•	B-<Entity>: Beginning of an entity
•	I-<Entity>: Inside of an entity
•	O: Outside of an entity
Please extract all chemicals, genes, and diseases mentioned in the paragraph. Provide the output in the format <word> - <BIO tag>, where each word is followed by its corresponding BIO tag.
''' + text for text in train_texts]

# # 对标签进行编码
# train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
# train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt", max_length=128)

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

# train_dataset = Dataset(train_encodings, train_labels_encodings)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

# 定义微调参数和优化器
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)

# # 开始微调
# epochs = 1
# # 训练模型
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
# torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/glm_bc2_loramodel.pt")

# 导入LoRA模型参数
base_model = AutoModelForCausalLM.from_pretrained(
    '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

lora_model = get_peft_model(base_model, lora_config)
lora_model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/glm_bc2_loramodel.pt"))

# 将模型移动到合适的设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lora_model.to(device)

# 读取测试数据
with open("./outputonebc2.txt", "r") as file:
    test_data = file.readlines()

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained('/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat', trust_remote_code=True)


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
with open("glm_bc2_lora.txt", "w") as output_file:
    for line in test_data:
        if line.strip():  # 跳过空行
            # 准备输入文本
            input_text = '''Generate BIO tags for each word in the given paragraph. The BIO format uses the following labels:
• B-<Entity>: Beginning of an entity
• I-<Entity>: Inside of an entity
• O: Outside of an entity
Please extract all chemicals, genes, and diseases mentioned in the paragraph. Provide the output in the format <word> - <BIO tag>, where each word is followed by its corresponding BIO tag.
''' + line.strip()
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            print(input_ids)
            outputs = lora_model.generate(input_ids, max_new_tokens=50)
            predicted_label = tokenizer.batch_decode(outputs)[0].strip()
            # 写入预测结果到文件
            output_file.write(f"{line.strip()}\t{predicted_label}\n")

