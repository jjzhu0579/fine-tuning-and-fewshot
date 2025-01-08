import json
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType

# Set environment variable for HF endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

prompt = '''Please answer the following QUESTION with yes or no according to the CONTEXT: '''

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
    data1 = []
    texts = []
    labels = []
    count = 0
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                count = count + 1
                question = item["questions"]
                context = item["context"]
                answer = item["answer"]
                input_text = "\t".join(question+' '+context)
                output_text = "\t".join(answer)
                data1.append({"input": input_text, "output": output_text})
                # Each label is separated by a space
    print(count)
    return pd.DataFrame(data1)

if __name__ == '__main__':
    # Load and process data
    data_path = "./Task7B_yesno_train.json"
    df = load_and_process_data(data_path)
    ds = Dataset.from_pandas(df)

    glm4_model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'
    lora_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/GLM4_qlora_bqa'

    tokenizer = AutoTokenizer.from_pretrained(glm4_model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(glm4_model_path, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.enable_input_require_grads()

    # LoRA configuration
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # 现存问题只微调部分演示即可
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
        num_train_epochs=5,
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
