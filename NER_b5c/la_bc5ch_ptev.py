import os
from datasets import load_dataset
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json

# 设置HF镜像地址
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

prompt = """
NER Extraction
Input: <sentence>
Please extract all entities mentioned in the sentence about chemicals/genes/diseases.
Answer in IOB format commonly used in NER tasks.
BIO tagging scheme is used in this task.
The Beginning and Inside in the IOB format need to be followed by a specific entity type.
"""

def process_val_dataset():
    dataset = []
    ds = load_dataset("bigbio/blurb", "bc5chem")
    for i in range(len(ds['test'])):
        line = ds['test'][i]
        # 将tokens拼接成字符串
        json_object = " ".join(line['tokens'])
        dataset.append(json_object)
    # 写入JSON文件
    with open('./test_result.json', "w", encoding="utf-8") as file:
        file.write(json.dumps(dataset, ensure_ascii=False))
    return dataset

if __name__ == '__main__':
    l3_model_path = "/data/aim_nuist/aim_zhujj/llama3"
    lora_path = '/data/aim_nuist/aim_zhujj/llama3/l3_pt_b5c'

    with open("test_result.json", "r") as f:
        val_list = json.load(f)

    model_id = "/data/aim_nuist/aim_zhujj/llama3"
    # 初始化pipeline
    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )

    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to('cuda')
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    # 遍历val_list
    for val in val_list:
        # 生成输入文本
        input_text = prompt.replace("<sentence>", val)

        # 使用pipeline生成结果
        outputs = pipe(
            input_text,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        # 获取生成的文本
        assistant_response = outputs[0]["generated_text"]

        # 写入结果文件
        with open('l3_pt_bc5ch_result.txt', 'a', encoding='utf-8') as f:
            f.write(f"{assistant_response}\n")
