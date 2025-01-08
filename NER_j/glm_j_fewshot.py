import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# 定义不同配置的少量示例
FEW_SHOT_EXAMPLES = {
    "0shot": "",
    "1shot": (
        'IgE    B-protein\n'
    ),
    "2shot": (
        'IgE    B-protein\n'
        'Th2    B-cell_line\n'
    ),
    "3shot": (
        'Specific    O\n'
        'IgE    B-protein\n'
        'Th2    B-cell_line\n'
    ),
    "5shot": (
        'cytokines    B-protein\n'
        '.O\n'
        'Specific    O\n'
        'IgE    B-protein\n'
        'Th2    B-cell_line\n'
    )
}

prompt = '''Generate BIO tags for each word in the given paragraph. The BIO format uses the following labels:
• B-protein: Beginning of a protein name
• I-protein: Inside of a protein name
• B-cell_type: Beginning of a cell type name
• I-cell_type: Inside of a cell type name
• B-DNA: Beginning of a DNA mention
• I-DNA: Inside of a DNA mention
• O: Outside of any entity
Please extract all chemicals, genes, and diseases mentioned in the paragraph. Provide the output in the format <word> - <BIO tag>, where each word is followed by its corresponding BIO tag.
'''


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
    test_file_path = './100fewshot.txt'
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
                    labels_to_find = ['B-protein', 'I-protein', 'O', 'B-cell_type', 'I-cell_type', 'B-GENE', 'I-GENE']

                    # 初始化一个变量保存最先找到的标签及其索引位置
                    earliest_label = None
                    earliest_index = len(generated_labels)  # 初始化为一个较大的值

                    # 遍历查找每个标签的位置
                    for label in labels_to_find:
                        index = generated_labels.find(label)
                        if 0 <= index < earliest_index:  # 如果找到并且位置比之前的更靠前
                            earliest_index = index
                            earliest_label = label

                    # 如果找到匹配的标签，替换 generated_labels 为最先找到的标签
                    if earliest_label:
                        generated_labels = earliest_label

                    # 输出结果
                    print(generated_labels)

                    # Combine the original input and generated labels
                    result = f"{test}\t{generated_labels}\n"

                    # Write results to respective output file
                    output_file = f'glm_bio_res_{shot}.txt'
                    with open(output_file, 'a', encoding='utf-8') as f:
                        print(result)
                        f.write(result)
            except Exception as e:
                print(f"Error processing test case: {test}\nError: {e}")
