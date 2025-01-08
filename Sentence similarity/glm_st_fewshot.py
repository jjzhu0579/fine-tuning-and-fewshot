import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# 定义不同配置的少量示例
FEW_SHOT_EXAMPLES = {
    "0shot": "",
    "1shot": "Sentence A: The cat is on the mat. Sentence B: The cat is sitting on the mat. Similar.\n",
    "2shot": (
        "Sentence A: The cat is on the mat. Sentence B: The cat is sitting on the mat. Similar.\n"
        "Sentence A: The dog barks loudly. Sentence B: The dog is silent. Different.\n"
    ),
    "3shot": (
        "Sentence A: The cat is on the mat. Sentence B: The cat is sitting on the mat. Similar.\n"
        "Sentence A: The dog barks loudly. Sentence B: The dog is silent. Different.\n"
        "Sentence A: The sun is bright today. Sentence B: The sun is shining. Similar.\n"
    ),
    "5shot": (
        "Sentence A: The cat is on the mat. Sentence B: The cat is sitting on the mat. Similar.\n"
        "Sentence A: The dog barks loudly. Sentence B: The dog is silent. Different.\n"
        "Sentence A: The sun is bright today. Sentence B: The sun is shining. Similar.\n"
        "Sentence A: He runs fast. Sentence B: She jogs slowly. Different.\n"
        "Sentence A: The food is delicious. Sentence B: The meal tastes great. Similar.\n"
    )
}

prompt = '''Please understand whether the two sentences are similar or different,just answer similar or different and no reason'''



def load_test_dataset(file_path):
    data = []
    skipped_lines = 0
    print(f"Reading file from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(line)
                else:
                    skipped_lines += 1
        print(f"Skipped {skipped_lines} empty lines.")
        print(f"Loaded {len(data)} valid test samples.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return data


if __name__ == '__main__':
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'

    # Load the test dataset from the local file
    test_file_path = './short_test.csv'
    test_list = load_test_dataset(test_file_path)

    # Debugging output
    print(f"Loaded {len(test_list)} test samples.")
    if test_list:
        print(f"Sample test data: {test_list[0]}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()

    # Iterate through each few-shot configuration
    for shot in ["0shot", "1shot", "2shot", "3shot", "5shot"]:
        full_prompt = prompt + "\n" + FEW_SHOT_EXAMPLES[shot]  # Combine base prompt with few-shot examples

        # Iterate through test_list
        for test in test_list:
            try:
                # Adjusted input tokenization
                inputs = tokenizer(
                    full_prompt + "\n" + test,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2500
                ).to('cuda')

                gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]

                    # Decode generated labels
                    generated_labels = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Combine the original input and generated labels
                    result = f"{test}\t{generated_labels}\n"

                    # Write results to respective output file
                    output_file = f'glm_similarity_res_{shot}.txt'
                    with open(output_file, 'a', encoding='utf-8') as f:
                        print(result)
                        f.write(result)
            except Exception as e:
                print(f"Error processing test case: {test}\nError: {e}")
