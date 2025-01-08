import json
import os

import pandas as pd

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import DataCollatorForSeq2Seq, Trainer
from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType,LoraConfig

prompt = """
NER Extraction
Input: <sentence>
Please extract all entities mentioned in the sentence about chemicals/genes/diseases.
Answer in IOB format commonly used in NER tasks.
BIO tagging scheme is used in this task.
The Beginning and Inside in the IOB format need to be followed by a specific entity type.
"""


def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer((f"[gMASK]<sop><|system|>\n{prompt}<|user|>\n"
                             f"{example['input']}<|assistant|>\n"
                             ).strip(),
                            add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def generate_ner_tags_label(tokens: list, tag: list):
    """
    生成ner_tags的label，'0': O，'1': B-GENE，'2': I-GENE
    :param tag:
    :return:
    """
    result = []
    # for遍历tokens，根据tag的值，生成对应的label，格式 token \t tag
    index = 0
    for t in tag:
        tmp = 'O'
        if t == 0:
            tmp = 'O'
        elif t == 1:
            tmp = 'B-GENE'
        elif t == 2:
            tmp = 'I-GENE'
        result.append(tokens[index] + '\t' + tmp)
        index += 1
    return result


if __name__ == '__main__':
    # ds = load_dataset("bigbio/blurb", "bc5disease")

    output_json = "./dataset_bc5dis_train.json"
    # output_json_objects = []
    # # 处理数据，遍历数据集当中的每一条记录
    # for i in range(len(ds['train'])):
    #     line = ds['train'][i]
    #     # line的tokens按空格拼接成字符串，作为dict的input的value，line的ner_tags按空格拼接成字符串，作为dict的output的value
    #     json_object = {"input": " ".join(line['tokens']),
    #                    "output": "\n".join(generate_ner_tags_label(line['tokens'], line['ner_tags'])) + "\n"}
    #     output_json_objects.append(json_object)
    #
    # # 保存数据集
    # with open(output_json, "w", encoding="utf-8") as file:
    #     file.write(json.dumps(output_json_objects, ensure_ascii=False) + "\n")

    df = pd.read_json(output_json)
    ds = Dataset.from_pandas(df)

    l3_model_path = "/data/aim_nuist/aim_zhujj/llama3"
    lora_path = '/data/aim_nuist/aim_zhujj/llama3/l3_lora_b5d'

    tokenizer = AutoTokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/llama3")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)


    def is_valid_sample(example):
        return len(example['input_ids']) > 0 and len(example['attention_mask']) > 0 and len(
            example['labels']) > 0 and len(example['input_ids']) == len(example['labels'])


    valid_tokenized_id = tokenized_id.filter(is_valid_sample)

    print(f"Original dataset size: {len(tokenized_id)}")
    print(f"Filtered dataset size: {len(valid_tokenized_id)}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModelForCausalLM.from_pretrained("/data/aim_nuist/aim_zhujj/llama3")
    model.enable_input_require_grads()  # 开启梯度检查点

    #  loraConfig
    config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
        encoder_dropout=0.1, encoder_num_layers=8, encoder_hidden_size=2048)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 配置训练参数
    args = TrainingArguments(
        output_dir="./output/GLM4_ql_b5c",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=50,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=5e-5,
        save_on_each_node=True,
        # gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=valid_tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    peft_model_id = lora_path
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

