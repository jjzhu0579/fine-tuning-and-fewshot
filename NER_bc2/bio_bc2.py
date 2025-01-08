import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
# 加载训练数据
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
# 初始化 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/BioClinicalBERT")
model=AutoModelForMaskedLM.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/BioClinicalBERT").to('cuda')

# 构建微调任务
train_texts = ['''Generate BIO tags for each word in the given paragraph,. The BIO format uses the following labels:
•	B-<Entity>: Beginning of an entity
•	I-<Entity>: Inside of an entity
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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 开始微调
epochs=2
# 训练模型
train_losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
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

    train_losses.append(epoch_loss / len(train_loader))

# 绘制损失曲线
plt.plot(np.arange(1, epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("training_loss_curve.png")

# 保存模型参数
torch.save(model.state_dict(), "bio_bc2_model.pt")

# 导入模型参数
model=AutoModelForMaskedLM.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/BioClinicalBERT").to('cuda')
model.load_state_dict(torch.load("bio_bc2_model.pt"))

# 将模型移动到合适的设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 读取测试数据
with open("./final_test.txt", "r") as file:
    test_data = file.readlines()

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/BioClinicalBERT")

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

with open("bio_bc2_result.txt", "w") as output_file:
    for line in test_data:
        if line.strip():  # Skip empty lines
            input_text = '''Generate BIO tags for each word in the given paragraph,. The BIO format uses the following labels:
•	B-<Entity>: Beginning of an entity
•	I-<Entity>: Inside of an entity
•	O: Outside of an entity
Please extract all chemicals, genes, and diseases mentioned in the paragraph. Provide the output in the format <word> - <BIO tag>, where each word is followed by its corresponding BIO tag.
''' + line.strip()  # Generate input for the model with each line
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            tokenized = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(device)
            output = model(**tokenized, return_dict=True)
            predicted_labels = output.logits[0].argmax(dim=-1)  # Get the indices of the highest probabilities
            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_labels)  # Convert IDs to tokens

            # Check if the 2nd token is "O"
            if predicted_tokens[1] == "o":
                label_text = predicted_tokens[1]
            elif predicted_tokens[1] == "b":
                label_tokens = predicted_tokens[1:5]  # Take the 2nd, 3rd, and 4th elements
                label_text = ' '.join(label_tokens)
                label_text = label_text.replace("#", "")
            elif predicted_tokens[1] == "i":
                label_tokens = predicted_tokens[1:5]  # Take the 2nd, 3rd, and 4th elements
                label_text = ' '.join(label_tokens)
                label_text = label_text.replace("#", "")
            else:
                label_tokens = predicted_tokens[1:4]  # Take the 2nd, 3rd, and 4th elements
                label_text = ' '.join(label_tokens)

            output_file.write(f"{line.strip()}\t{label_text}\n")  # Write the labeled line to output.txt
