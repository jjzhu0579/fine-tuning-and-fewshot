import os

from datasets import load_dataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = """
NER Extraction
Input: <sentence>
Please extract all entities mentioned in the sentence about chemicals/genes/diseases.
Answer in IOB format commonly used in NER tasks.
BIO tagging scheme is used in this task.
The Beginning and Inside in the IOB format need to be followed by a specific entity type.
"""


import json
def process_val_dataset():
    dataset = []
    ds = load_dataset("bigbio/blurb", "bc5chem")
    for i in range(len(ds['test'])):
        line = ds['test'][i]
        # line的tokens按空格拼接成字符串，作为dict的input的value，line的ner_tags按空格拼接成字符串，作为dict的output的value
        json_object = " ".join(line['tokens'])
        dataset.append(json_object)
        with open('./test_result.json', "w", encoding="utf-8") as file:
            file.write(json.dumps(dataset, ensure_ascii=False))
    return dataset


if __name__ == '__main__':
    l3_model_path = "/data/aim_nuist/aim_zhujj/llama3"
    lora_path = '/data/aim_nuist/aim_zhujj/llama3/l3_pt_b2c'

    with open("testbc2_result.json", "r") as f:
        val_list = json.load(f)


    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/llama3")
    tokenizer.pad_token = tokenizer.eos_token
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained("/data/aim_nuist/aim_zhujj/llama3").to('cuda')
    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    # 遍历val_list
    for val in val_list:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "user", "content": val}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to('cuda')

        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(inputs, **gen_kwargs)
            out = tokenizer.batch_decode(outputs)[0]
            # 打开文件output.txt，将输出结果写入文件
            with open('l3_pt_bc2_result.txt', 'a', encoding='utf-8') as f:
                f.write(f"{out}\n")


