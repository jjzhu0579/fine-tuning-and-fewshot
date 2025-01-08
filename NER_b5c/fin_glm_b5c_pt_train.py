import json
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, Trainer
from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType

# Set environment variable for HF endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Define the prompt
prompt = """
please select labels from the following range according to the text : O,I-Disease,B-Disease
"""

# Function to process the dataset
def process_func(example):
    MAX_LENGTH = 384
    instruction = tokenizer(
        (f"[gMASK]<sop>\n{prompt}\n{example['input']}\n").strip(),
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Function to load and process data from final_train.txt
def load_and_process_data(file_path):
    data = []
    skipped_lines = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        tokens = []
        labels = []
        for line in file:
            line = line.strip()
            if line:
                try:
                    token, label = line.split('\t')
                    tokens.append(token)
                    labels.append(label)
                except ValueError:
                    skipped_lines += 1
            else:
                if tokens and labels:
                    input_text = " ".join(tokens)
                    output_text = " ".join(labels)  # Each label is separated by a space
                    data.append({"input": input_text, "output": output_text})
                    tokens, labels = [], []

    print(f"Skipped {skipped_lines} lines that didn't have exactly two values.")
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Load and process data
    data_path = "./train.tsv.txt"
    df = load_and_process_data(data_path)
    ds = Dataset.from_pandas(df)

    glm4_model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'
    lora_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/GLM4_pt_b5c'

    tokenizer = AutoTokenizer.from_pretrained(glm4_model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(glm4_model_path, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.enable_input_require_grads()

    # LoRA configuration
    config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
        encoder_dropout=0.1, encoder_num_layers=5, encoder_hidden_size=1024
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Training arguments
    args = TrainingArguments(
        output_dir="./output/GLM4",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=50,
        num_train_epochs=10,
        save_steps=100,
        learning_rate=1e-5,
        save_on_each_node=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    peft_model_id = lora_path
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
