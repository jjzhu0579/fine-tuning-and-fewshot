import torch
from peft import PeftModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import os

# 定义不同配置的少量示例
FEW_SHOT_EXAMPLES = {
    "0shot": "",
    "1shot": "Context: This is an example context. Question: Is this true? Yes.\n",
    "2shot": (
        "Context: This is an example context. Question: Is this true? Yes.\n"
        "Context: Another context for a different question. Question: Is this correct? No.\n"
    ),
    "3shot": (
        "Context: This is an example context. Question: Is this true? Yes.\n"
        "Context: Another context for a different question. Question: Is this correct? No.\n"
        "Context: This context relates to a third question. Question: Is this valid? Yes.\n"
    ),
    "5shot": (
        "Context: This is an example context. Question: Is this true? Yes.\n"
        "Context: Another context for a different question. Question: Is this correct? No.\n"
        "Context: This context relates to a third question. Question: Is this valid? Yes.\n"
        "Context: This is yet another context. Question: Is this relevant? No.\n"
        "Context: Final example context. Question: Is this acceptable? Yes.\n"
    )
}

prompt = '''please select labels from the following range according to the text : yes,no'''


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

    # Load the test dataset from the local file
    test_file_path = 'test_set.txt'
    test_list = load_test_dataset(test_file_path)

    # Debugging output
    print(f"Loaded {len(test_list)} test samples.")
    if test_list:
        print(f"Sample test data: {test_list[0]}")

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")
    model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # Iterate through each few-shot configuration
    for shot in ["0shot", "1shot", "2shot", "3shot", "5shot"]:
        full_prompt = prompt + "\n" + FEW_SHOT_EXAMPLES[shot]  # Combine base prompt with few-shot examples

        for test in test_list:
            try:
                # Adjusted input tokenization
                input_text = full_prompt + " : " + test

                gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

                with torch.no_grad():
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                    outputs = model.generate(input_ids)
                    predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Combine the generated labels with the result format
                    result = f"thenewline : {predicted_label}\n"

                    # Write results to respective output file
                    output_file = f't5_qa_fewshot_res_{shot}.txt'
                    with open(output_file, 'a', encoding='utf-8') as f:
                        print(result)
                        f.write(result)
            except Exception as e:
                print(f"Error processing test case: {test}\nError: {e}")
