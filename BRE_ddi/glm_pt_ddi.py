import torch
import os
from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType
import matplotlib.pyplot as plt
import numpy as np
from peft import TaskType,LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
# 加载训练数据
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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
# 初始化 tokenizer 和模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 初始化 tokenizer 和模型
glm4_model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'
# tokenizer = AutoTokenizer.from_pretrained('/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat', trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(glm4_model_path, device_map="cuda:0", torch_dtype=torch.bfloat16,
                                            trust_remote_code=True)
# 配置和加载LoRA模型
config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10,
    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
    encoder_dropout=0.1, encoder_num_layers=5, encoder_hidden_size=1024
)
# model = get_peft_model(base_model, config)

# 构建微调任务
# train_texts = ['''Ask whether there is a relationship between the two entities of the sentence : ''' + text for text in train_texts]
#
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
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
#
# # 定义微调参数和优化器
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
#         print(type(outputs))
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
# torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/glm_ddi_qloramodel.pt")

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptEncoderConfig, TaskType, PromptEncoderReparameterizationType, get_peft_model

# 导入LoRA模型参数

# 配置和加载LoRA模型
config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10,
    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
    encoder_dropout=0.1, encoder_num_layers=5, encoder_hidden_size=1024
)
lora_model = get_peft_model(base_model, config)
lora_model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/glm_ddi_qloramodel.pt"))

# 将模型移动到合适的设备，并使用 DataParallel
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个 GPU 进行并行处理")
    lora_model = torch.nn.DataParallel(lora_model)
lora_model.to(device)

# 读取测试数据
with open("./changshi.txt", "r") as file:
    test_data = file.readlines()

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained('/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat', trust_remote_code=True)


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

from tqdm import tqdm  # 导入 tqdm 库

with open("gl_ddi_ptt.txt", "w") as output_file:


    # 使用 tqdm 创建测试循环的进度条
    with tqdm(test_data, desc="Testing", unit="line") as ttest:
        for i, line in enumerate(ttest):
            if line.strip():  # 跳过空行
                # 进行单词或标点的标签预测
                input_text = '''Ask whether there is a relationship between the two entities of the sentence: ''' + line.strip()
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                print(input_ids)
                outputs = lora_model.generate(input_ids=input_ids, bos_token_id=tokenizer.bos_token_id)
                predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 写入预测结果到output.txt
                output_file.write(f"{line.strip()}\t{predicted_label}\n")
