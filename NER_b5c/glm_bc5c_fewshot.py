import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# 定义基础提示
base_prompt = """Please select labels from the following range according to the text: O, I-Disease, B-Disease.\n"""

# 定义不同 shot 的 few-shot 示例
shot_examples = {
    1: """Text: "The patient has diabetes."
Answer: I-Disease\n""",

    2: """Text: "The patient has diabetes."
Answer: I-Disease
Text: "There is a report of hypertension."
Answer: B-Disease\n""",

    3: """Text: "The patient has diabetes."
Answer: I-Disease
Text: "There is a report of hypertension."
Answer: B-Disease
Text: "Symptoms include fatigue."
Answer: O\n""",

    5: """Text: "The patient has diabetes."
Answer: I-Disease
Text: "There is a report of hypertension."
Answer: B-Disease
Text: "Symptoms include fatigue."
Answer: O
Text: "Cholesterol levels were elevated."
Answer: I-Disease
Text: "The treatment plan was discussed."
Answer: O\n"""
}

def load_test_dataset(file_path):
    data = []
    skipped_lines = 0
    print(f"Reading file from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        token, _ = line.split('\t')
                        data.append(token)
                    except ValueError:
                        skipped_lines += 1
        print(f"Skipped {skipped_lines} lines that didn't have exactly two values.")
        print(f"Loaded {len(data)} valid test samples.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return data

def extract_label(generated_text):
    # 正则表达式匹配 'O'、'I-Disease' 或 'B-Disease'
    match = re.search(r'\b(O|I-Disease|B-Disease)\b', generated_text)
    return match.group(0) if match else "NoLabel"

def test_model(test_list, shot, model, tokenizer):
    # 创建带有相应 few-shot 示例的提示
    prompt = base_prompt + shot_examples.get(shot, "")

    output_file = f'glm_b5c_pt_res_{shot}shot.txt'  # 为每个 shot 单独保存结果

    for test in test_list:
        try:
            # 创建带有测试样例的输入
            inputs = tokenizer(
                prompt + "Text: " + test + "\nAnswer:",
                return_tensors="pt",
                truncation=True,
                max_length=2500
            ).to('cuda')

            gen_kwargs = {"max_length": 250, "do_sample": True, "top_k": 1}

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]

                # 解码生成结果
                generated_labels = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 从生成文本中提取第一个符合要求的标签 ('O', 'I-Disease', 'B-Disease')
                predicted_label = extract_label(generated_labels)

                # 将原始输入和预测标签组合
                result = f"{test}\t{predicted_label}\n"

                # 将结果写入相应的输出文件
                with open(output_file, 'a', encoding='utf-8') as f:
                    print(result)
                    f.write(result)
        except Exception as e:
            print(f"Error processing test case: {test}\nError: {e}")

if __name__ == '__main__':
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'

    # 加载测试数据集
    test_file_path = 'testthree.txt'
    test_list = load_test_dataset(test_file_path)

    # 调试输出
    print(f"Loaded {len(test_list)} test samples.")
    if test_list:
        print(f"Sample test data: {test_list[0]}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()

    # 对每个 shot 场景进行测试 (0-shot, 1-shot, 2-shot, 3-shot, 5-shot)
    for shot in [0, 1, 2, 3, 5]:
        test_model(test_list, shot, model, tokenizer)
