import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss

def custom_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Ensure input_ids and position_ids are Long type
    if input_ids is not None:
        input_ids = input_ids.to(torch.long)
    if position_ids is not None:
        position_ids = position_ids.to(torch.long)

    # Ensure attention_mask and inputs_embeds are float
    if attention_mask is not None:
        attention_mask = attention_mask.to(torch.float32)
    if inputs_embeds is not None:
        inputs_embeds = inputs_embeds.to(torch.float32)

    transformer_outputs = self.transformer(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = transformer_outputs[0]
    if return_last_logit:
        hidden_states = hidden_states[:, -1:]
    lm_logits = self.transformer.output_layer(hidden_states)

    loss = None
    if labels is not None:
        lm_logits = lm_logits.to(torch.float32)
        labels = labels.to(torch.int64)  # CrossEntropyLoss expects int64 labels

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)

        # Ensure the loss is a floating point tensor and requires gradients
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.to(hidden_states.dtype).requires_grad_(True)

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )

# 启用离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 读取训练数据
with open('./final_train.txt', 'r') as file:
    train_data = file.readlines()
train_texts = []
train_labels = []
invalid_lines_count = 0

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

print(f"Number of invalid lines: {invalid_lines_count}")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat',
                                          trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat',
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(device).eval()

config = PromptEncoderConfig(
    task_type=TaskType.TOKEN_CLS, num_virtual_tokens=10,
    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
    encoder_dropout=0.1, encoder_num_layers=4, encoder_hidden_size=4096)
model = get_peft_model(base_model, config)
model.forward = custom_forward.__get__(model)

# 构建微调任务
train_texts = ['''Generate BIO tags for each word in the given paragraph,. The BIO format uses the following labels:
•    B: Beginning of an entity
•    I: Inside of an entity
•    O: Outside of an entity
Please extract all chemicals, genes, and diseases mentioned in the paragraph. Provide the output in the format <word> - <BIO tag>, where each word is followed by its corresponding BIO tag.
''' + text for text in train_texts]

train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=256)
train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt", max_length=256)

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

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
model.to(device)

# Training Loop with tqdm
# epochs = 1
# train_losses = []
#
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0
#
#     # Use tqdm to wrap train_loader
#     for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
#         optimizer.zero_grad()
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)
#         max_seq_length = input_ids.shape[1]
#         padded_labels = torch.nn.functional.pad(labels, (0, max_seq_length - labels.shape[1]), value=-100).to(device)
#
#         try:
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=padded_labels)
#             loss = outputs.loss
#         except Exception as e:
#             print(f"Error during model forward pass: {e}")
#             raise
#
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#
#     train_losses.append(epoch_loss / len(train_loader))
#
# # Plotting the training loss
# plt.plot(np.arange(1, epochs + 1), train_losses, label="Training Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training Loss Curve")
# plt.legend()
# plt.savefig("training_loss_curve.png")
#
# torch.save(model.state_dict(), "/data/aim_nuist/aim_zhujj/xinjian/glm_bc2_pt_model.pt")

model = AutoModelForCausalLM.from_pretrained(
    '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora/ZhipuAI/glm-4-9b-chat',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

model = get_peft_model(model, config)
model.load_state_dict(torch.load("/data/aim_nuist/aim_zhujj/xinjian/glm_bc2_pt_model.pt"))
model.to(device)

from tqdm import tqdm

with open("./ttt.txt", "r") as file:
    test_data = file.readlines()

# CustomTestDataset class
class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx].strip()
        encoding = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=256)
        # Squeeze to remove the batch dimension added by the tokenizer
        return {key: val.squeeze(0) for key, val in encoding.items()}

test_dataset = CustomTestDataset(test_data, tokenizer)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
outputs = []

# Adding tqdm to monitor the progress
for batch in tqdm(test_loader, desc="Testing"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        outputs.append(model.generate(input_ids=input_ids, attention_mask=attention_mask))

output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
with open("./test_outputs.txt", "w") as file:
    for text in output_texts:
        file.write(text + "\n")
