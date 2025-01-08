import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np

# 加载训练数据
with open('/share/home/aim/aim_zhujj/FLAN-T5/test.iob2', 'r') as file:
    train_data = file.readlines()

train_texts = []
train_labels = []

for line in train_data:
    if line.strip():
        word, label = line.strip().split("\t")
        if len(word) == 1 and not word.isalnum():
            train_texts.append(word)
            train_labels.append("O")
        else:
            train_texts.append(word)
            train_labels.append(label)

from sklearn.model_selection import train_test_split
model_name = '/share/home/aim/aim_zhujj/PubMedBERT'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda')

# Split the dataset into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# Tokenize and pad the training and testing data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
train_labels_encodings = tokenizer(train_labels, padding=True, return_tensors="pt", max_length=128, truncation=True)

test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
test_labels_encodings = tokenizer(test_labels, padding=True, return_tensors="pt", max_length=128, truncation=True)

# Define Dataset and DataLoader for training and testing
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels_encodings):
        self.encodings = encodings
        self.labels_encodings = labels_encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels_encodings['input_ids'][idx]  # Remove torch.tensor() conversion
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = Dataset(train_encodings, train_labels_encodings)
test_dataset = Dataset(test_encodings, test_labels_encodings)


def custom_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 对 input_ids、attention_mask 和 labels 进行填充
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


# 使用自定义的 collate_fn 定义 DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

epochs = 5
train_losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # Pad the labels tensor to match the maximum sequence length
        max_seq_length = input_ids.shape[1]
        padded_labels = torch.nn.functional.pad(labels, (0, max_seq_length - labels.shape[1]))

        # Check the shape of the padded labels tensor
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=padded_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

# Evaluation on the test set
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        test_loss += outputs.loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Average Test Loss: {avg_test_loss}")

# Plotting the training loss curve
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('training_loss_curve.png')

# Save the model parameters
torch.save(model.state_dict(), 'pubmedbert_model.pt')