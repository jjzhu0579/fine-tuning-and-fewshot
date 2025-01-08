import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import prompt


def process_val_dataset(file_path: str):
    dataset = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        word_list = ""
        for line in lines:
            line = line.strip()
            if line:  # 非空行
                word = line
                word_list += word + " "
            else:
                dataset.append(word_list)
                word_list = ""
    return dataset


if __name__ == '__main__':
    mode_path = './ZhipuAI/glm-4-9b-chat'
    lora_path = './GLM4_lora'
    val_dataset_path = './sampletest3.iob2'

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()

    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    val_list = process_val_dataset(val_dataset_path)

    # 遍历val_list
    for val in val_list:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "user", "content": val}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to('cuda')

        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            # 打开文件output.txt，将输出结果写入文件
            with open('output.txt', 'a', encoding='utf-8') as f:
                result = tokenizer.decode(outputs[0], skip_special_tokens=True) + '\n'
                print(result)
                f.write(result)

