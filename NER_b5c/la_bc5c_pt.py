import os
import json
import pandas as pd
from datasets import Dataset
import torch
from transformers import pipeline, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

prompt = """
NER Extraction
Input: <sentence>
Please extract all entities mentioned in the sentence about chemicals/genes/diseases.
Answer in IOB format commonly used in NER tasks.
BIO tagging scheme is used in this task.
The Beginning and Inside in the IOB format need to be followed by a specific entity type.
"""

# Define the pipeline
model_id = "/data/aim_nuist/aim_zhujj/llama3"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# Function to generate NER tags from the output
def generate_ner_tags_label(tokens: list, tag: list):
    """
    Generate NER tags in 'O', 'B-GENE', 'I-GENE' format.
    """
    result = []
    for t, token in zip(tag, tokens):
        label = 'O'
        if t == 1:
            label = 'B-GENE'
        elif t == 2:
            label = 'I-GENE'
        result.append(f"{token}\t{label}")
    return result

# Load dataset
output_json = "./dataset_bc5ch_train.json"
df = pd.read_json(output_json)
ds = Dataset.from_pandas(df)

# Process function for dataset
def process_func(example):
    input_text = prompt.replace("<sentence>", example['input'])
    output = pipe(
        input_text,
        max_new_tokens=256,
        eos_token_id=pipe.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response_text = output[0]["generated_text"]
    response_tokens = response_text.split()
    ner_tags = generate_ner_tags_label(response_tokens, example['output'])

    return {
        "input_ids": pipe.tokenizer(input_text, add_special_tokens=False)["input_ids"],
        "attention_mask": pipe.tokenizer(input_text, add_special_tokens=False)["attention_mask"],
        "labels": pipe.tokenizer(ner_tags, add_special_tokens=False)["input_ids"]
    }

# Tokenize the dataset
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# Filter valid samples
def is_valid_sample(example):
    return len(example['input_ids']) > 0 and len(example['attention_mask']) > 0 and len(
        example['labels']) > 0 and len(example['input_ids']) == len(example['labels'])

valid_tokenized_id = tokenized_id.filter(is_valid_sample)

print(f"Original dataset size: {len(tokenized_id)}")
print(f"Filtered dataset size: {len(valid_tokenized_id)}")

# Model configuration
config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM, num_virtual_tokens=40,
    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
    encoder_dropout=0.1, encoder_num_layers=32, encoder_hidden_size=8192
)

# Load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_peft_model(pipe.model, config)
model.print_trainable_parameters()

# Training arguments
args = TrainingArguments(
    output_dir="./output/GLM4_pt_b5c",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    logging_steps=50,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=5e-5,
    save_on_each_node=True,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=valid_tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=pipe.tokenizer, padding=True),
)

# Train the model
trainer.train()

# Save the model and tokenizer
lora_path = '/data/aim_nuist/aim_zhujj/llama3/l3_pt_b5c'  # Define the path for saving the LoRA fine-tuned model
trainer.model.save_pretrained(lora_path)
pipe.tokenizer.save_pretrained(lora_path)
