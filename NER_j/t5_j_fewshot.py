import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
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
    test_file_path = '/data/aim_nuist/aim_zhujj/ner/sampletest3.iob2'
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

        # Iterate through test_list
        for test in test_list:
            try:
                # Adjusted input tokenization
                input_text = full_prompt + "\n" + test

                gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

                with torch.no_grad():
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                    outputs = model.generate(input_ids)
                    predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Combine the original input and generated labels
                    result = f"{test}\t{predicted_label}\n"

                    # Write results to respective output file
                    output_file = f't5_bio_res_{shot}.txt'
                    with open(output_file, 'a', encoding='utf-8') as f:
                        print(result)
                        f.write(result)
            except Exception as e:
                print(f"Error processing test case: {test}\nError: {e}")
