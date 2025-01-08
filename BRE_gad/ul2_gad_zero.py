import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
prompt = """
please select labels from the following range according to the text : true,false
"""


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


if __name__ == '__main__':
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/ul2'


    # Load the test dataset from the local file
    test_file_path = 'test1.tsv'
    test_list = load_test_dataset(test_file_path)

    # Debugging output
    print(f"Loaded {len(test_list)} test samples.")
    if test_list:
        print(f"Sample test data: {test_list[0]}")

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                       trust_remote_code=True).eval()

    # Iterate through test_list
    # Iterate through test_list
    for test in test_list:
        try:
            # Adjusted input tokenization
            input_text = prompt + " : " + test

            gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

            with torch.no_grad():
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                outputs = model.generate(input_ids)
                predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 获取原始输入文本
                original_text = test  # 使用test作为原始输入文本

                # 将原始文本和生成的标签组合成指定格式
                result = f"{original_text}\t{predicted_label}\n"

                # 将结果写入文件 output.txt
                with open('ul2_gad_zero_res.txt', 'a', encoding='utf-8') as f:
                    print(result)
                    f.write(result)
        except Exception as e:
            print(f"Error processing test case: {test}\nError: {e}")

