import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# Load training data
with open('./output_file.csv', 'r') as file:
    train_data = file.readlines()
train_texts = []
train_labels = []

invalid_lines_count = 0

for line in train_data:
    if line.strip():
        parts = line.strip().split("\t")
        # Check if there are two parts
        if len(parts) == 2:
            word, label = parts
            if len(word) == 1 and not word.isalnum():
                train_texts.append(word)
                train_labels.append("O")
            else:
                train_texts.append(word)
                train_labels.append(label)
        else:
            # Count invalid lines
            invalid_lines_count += 1

# Print number of invalid lines
print(f"Number of invalid lines: {invalid_lines_count}")

# Initialize tokenizer and model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat', trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

# Configure and load LoRA model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(base_model, lora_config)

# Prepare training texts
train_texts = ['''Please understand whether the two sentences are similar or different''' + text for text in train_texts]

# Encode labels and texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt", max_length=128)

# Define dataset class
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

# Define training parameters and optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.to(device)

# Training loop with progress bar
# epochs = 200
# train_losses = []
#
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0
#
#     with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
#         for batch in train_loader:
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
#             # Update the progress bar
#             pbar.update(1)
#             pbar.set_postfix({"Loss": loss.item()})
#
#     train_losses.append(epoch_loss / len(train_loader))
#     print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss / len(train_loader)}")
#
# # Plot training loss curve
# plt.plot(np.arange(1, epochs + 1), train_losses, label="Training Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training Loss Curve")
# plt.legend()
# plt.savefig("training_loss_curve.png")
#
# # Save model parameters
# torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/glmlo_st_model.pt")

# Load model parameters for testing
model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/glmlo_st_model.pt"))

# Load test data
with open("./short_test.csv", "r") as file:
    test_data = file.readlines()

# Define custom test dataset class
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

# Create test data loader
test_dataset = CustomTestDataset(test_data, tokenizer)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

# Predict labels and output to a file with progress bar
with open("glmlo_st_result.txt", "w") as output_file:
    for i, line in enumerate(tqdm(test_data, desc="Testing", unit="line")):
        if line.strip():  # 跳过空行
            input_text = '''Please understand whether the two sentences are similar or different''' + line.strip()
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

            # 添加调试打印
            print(f"Processing line {i + 1}/{len(test_data)}: {input_text}")

            try:
                outputs = model.generate(input_ids=input_ids, max_length=128)
                predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 将结果写入输出文件
                output_file.write(f"{line.strip()}\t{predicted_label}\n")

            except Exception as e:
                print(f"Error during generation: {e}")
                break  # 出现错误时停止

