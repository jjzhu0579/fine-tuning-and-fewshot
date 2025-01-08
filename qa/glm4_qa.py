import os
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
# 加载数据集
def load_dataset(folder, filenames):
    texts = []
    labels = []
    for filename in filenames:
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for key, value in data.items():
                    question = value["QUESTION"]
                    contexts = " ".join(value["CONTEXTS"])
                    context_labels = " ".join(value["LABELS"])
                    meshes = " ".join(value["MESHES"])
                    reasoning_required = value["reasoning_required_pred"]
                    reasoning_free = value["reasoning_free_pred"]
                    long_answer = value["LONG_ANSWER"]
                    final_decision = value["final_decision"]

                    input_text = (
                        f"Please answer the QUESTION in conjunction with the CONTEXTS: {question} context: {contexts} "
                        f"Labels: {context_labels} Features: {meshes} Reasoning Required: {reasoning_required} "
                        f"Reasoning Free: {reasoning_free} Detailed Answer: {long_answer}")

                    texts.append(input_text)
                    labels.append(final_decision)
    return texts, labels


# 加载训练数据
train_texts, train_labels = load_dataset('./pqal_fold0',
                                         ['train_set.json', 'dev_set.json'])
for i in range(1, 10):
    folder = f'./pqal_fold{i}'
    texts, labels = load_dataset(folder, ['train_set.json', 'dev_set.json' ])
    train_texts.extend(texts)
    train_labels.extend(labels)

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1,
                                                                    random_state=42)
device = "cuda"
# 初始化 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained('/share/home/aim/aim_zhujj/glm4_lora/ZhipuAI/glm-4-9b-chat', trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    '/share/home/aim/aim_zhujj/glm4_lora/ZhipuAI/glm-4-9b-chat',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

# 配置和加载LoRA模型
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=0.1,  # 增加Dropout
    bias="none",
    task_type="TaskType.SEQ_2_SEQ_LM"
)
model = get_peft_model(base_model, lora_config)

# 构建训练和验证数据
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt", max_length=10)

val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
val_labels_encodings = tokenizer(val_labels, truncation=True, padding=True, return_tensors="pt", max_length=10)


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
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 训练循环
epochs = 5 # 训练50个周期
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
plt.savefig("loss_curve.png")

# 保存模型参数
torch.save(model.state_dict(), "glm_lora_model.pt")

# 加载模型参数用于推理
base_model = AutoModelForCausalLM.from_pretrained(
    '/share/home/aim/aim_zhujj/glm4_lora/ZhipuAI/glm-4-9b-chat',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()
lora_model = get_peft_model(base_model, lora_config)
lora_model.load_state_dict(torch.load("glm_lora_model.pt"))
lora_model.to(device)
# 读取测试数据
with open('test_set.json', 'r', encoding='utf-8') as file:
    test_data = json.load(file)

test_texts = []
test_ids = []
for key, value in test_data.items():
    question = value["QUESTION"]
    contexts = " ".join(value["CONTEXTS"])
    context_labels = " ".join(value["LABELS"])
    meshes = " ".join(value["MESHES"])
    reasoning_required = value["reasoning_required_pred"]
    reasoning_free = value["reasoning_free_pred"]
    long_answer = value["LONG_ANSWER"]

    input_text = (f"Please answer the QUESTION in conjunction with the CONTEXTS: {question} context: {contexts} "
                  f"Labels: {context_labels} Features: {meshes} Reasoning Required: {reasoning_required} "
                  f"Reasoning Free: {reasoning_free} Detailed Answer: {long_answer}")

    test_texts.append(input_text)
    test_ids.append(key)

test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


test_dataset = TestDataset(test_encodings)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 预测标签并输出结果
results = {}
# 过滤非ASCII字符的函数
def filter_non_ascii(text):
    return ''.join([char if ord(char) < 128 else '' for char in text])


def filter_non_ascii(text):
    return ''.join([char for char in text if ord(char) < 128])

def extract_last_yes_no_maybe(predicted_label):
    # 提取最后一个 "yes", "no", 或 "maybe"
    for word in reversed(predicted_label.split()):
        if word.lower() in ["yes", "no", "maybe"]:
            return word.lower()
    return ""
with open("lora_glm_qa.txt", "w", encoding='utf-8') as output_file:
    for i, batch in enumerate(test_loader):
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
            extracted_label = extract_last_yes_no_maybe(predicted_label)  # 提取最后的yes/no/maybe
            results[test_ids[i]] = extracted_label
            print(extracted_label)

        output_file.write(f'"{test_ids[i]}": "{extracted_label}",\n')

# 输出JSON格式
with open("lora_glm_qa.json", "w", encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)