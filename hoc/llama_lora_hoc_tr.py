import json
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType

# Set environment variable for HF endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# Function to load and clean label options from 'labels' folder
def load_and_clean_label_options(label_folder):
    labels_set = set()  # Use a set to ensure no duplicates
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    for file in label_files:
        file_path = os.path.join(label_folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            label = f.read().strip().lower()  # Convert to lowercase for comparison
            if label not in ['null', '']:  # Exclude 'null', 'NULL', or empty strings
                labels_set.add(label)

    return list(labels_set)  # Convert set back to list


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


# Function to load and process data from text and label folders
def load_and_process_data(text_folder, label_folder, label_set):
    data = []
    skipped_files = 0

    # Get all the text files in the text folder
    text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]

    for text_file in text_files:
        text_path = os.path.join(text_folder, text_file)
        label_path = os.path.join(label_folder, text_file)  # Expect label file with the same name

        # Check if the corresponding label file exists
        if not os.path.exists(label_path):
            print(f"Label file not found for {text_file}")
            skipped_files += 1
            continue

        # Load the text and label content
        with open(text_path, 'r', encoding='utf-8') as text_f:
            text_content = text_f.read().strip()

        with open(label_path, 'r', encoding='utf-8') as label_f:
            label_content = label_f.read().strip().lower()  # Convert to lowercase for comparison

        # Determine the label: true if in label_set, otherwise false
        if label_content in ['NULL', 'sustaining proliferative signaling', 'enabling replicative immortality','resisting cell death', 'inducing angiogenesis']:
            label = 'false'
        elif label_content in label_set:
            label = 'true'
        else:
            label = 'false'
        print(label)
        # Append the data as input-output pairs
        data.append({"input": text_content, "output": label})

    print(f"Skipped {skipped_files} files due to missing label files.")
    return pd.DataFrame(data)


if __name__ == '__main__':
    # Set paths to text and label folders
    text_folder = "./text"
    label_folder = "./labels"

    # Load and clean label options for the prompt
    label_options = load_and_clean_label_options(label_folder)
    print(label_options)
    label_options_str = ", ".join(label_options)
    # Define the prompt with a check for the label being in the set
    prompt = f"""
    Determine the label for the following text. If the label is 'null' or 'NULL', respond with 'false'. 
    If the label belongs to the following valid options [{label_options_str}], respond with 'true'. 
    """

    # Load and process data
    df = load_and_process_data(text_folder, label_folder, set(label_options))
    ds = Dataset.from_pandas(df)

    llama_model_path = "/data/aim_nuist/aim_zhujj/llama3"
    lora_path = '/data/aim_nuist/aim_zhujj/llama3/llama_pt_gad'

    tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(llama_model_path, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.enable_input_require_grads()

    # LoRA configuration
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1  # Dropout 比例
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Training arguments
    args = TrainingArguments(
        output_dir="./output/GLM4",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=50,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
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
