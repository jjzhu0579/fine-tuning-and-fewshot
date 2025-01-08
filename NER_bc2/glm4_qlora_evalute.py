import os

from datasets import load_dataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import prompt


def process_val_dataset():
    dataset = []
    ds = load_dataset("spyysalo/bc2gm_corpus")
    for i in range(len(ds['test'])):
        line = ds['test'][i]
        # line的tokens按空格拼接成字符串，作为dict的input的value，line的ner_tags按空格拼接成字符串，作为dict的output的value
        json_object = " ".join(line['tokens'])
        dataset.append(json_object)

    return dataset


if __name__ == '__main__':
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'
    lora_path = './GLM4_qlora'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    val_list = process_val_dataset()

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()

    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    # 遍历val_list
    for val in val_list:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "user", "content": val}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(device)

        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            # 打开文件output.txt，将输出结果写入文件
            with open('glm_ql_bc2.txt', 'a', encoding='utf-8') as f:
                f.write(tokenizer.decode(outputs[0], skip_special_tokens=True) + '\n')


