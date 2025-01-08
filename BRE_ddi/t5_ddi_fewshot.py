import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Base prompt for zero-shot
prompt_base = """
please select labels from the following range according to the text : true,false
"""

# Few-shot examples for 1-shot, 2-shot, 3-shot, 5-shot
few_shot_examples = [
    "Text: The patient showed significant improvement.\ttrue\n",
    "Text: The study failed to show any improvement.\tfalse\n",
    "Text: There was no significant change in the condition.\tfalse\n",
    "Text: The treatment reduced inflammation and pain.\ttrue\n",
    "Text: The outcome was inconclusive.\tfalse\n"
]


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


def generate_and_save_results(test_list, prompt, output_file, model, tokenizer):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Iterate through test_list
    for test in test_list:
        try:
            # Adjust input text with the few-shot prompt
            input_text = prompt + " : " + test

            with torch.no_grad():
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                outputs = model.generate(input_ids)
                predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Combine original text and generated label
                result = f"{test}\t{predicted_label}\n"

                # Save the result to the output file
                with open(output_file, 'a', encoding='utf-8') as f:
                    print(result)
                    f.write(result)
        except Exception as e:
            print(f"Error processing test case: {test}\nError: {e}")


if __name__ == '__main__':
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5'

    # Load the test dataset
    test_file_path = 'combined_test.txt'
    test_list = load_test_dataset(test_file_path)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                       trust_remote_code=True).eval()

    # Zero-shot
    zero_shot_prompt = prompt_base
    generate_and_save_results(test_list, zero_shot_prompt, 'flan-t5_ddi_zero_res.txt', model, tokenizer)

    # One-shot
    one_shot_prompt = prompt_base + "\n" + few_shot_examples[0]
    generate_and_save_results(test_list, one_shot_prompt, 'flan-t5_ddi_one_res.txt', model, tokenizer)

    # Two-shot
    two_shot_prompt = prompt_base + "\n" + few_shot_examples[0] + few_shot_examples[1]
    generate_and_save_results(test_list, two_shot_prompt, 'flan-t5_ddi_two_res.txt', model, tokenizer)

    # Three-shot
    three_shot_prompt = prompt_base + "\n" + few_shot_examples[0] + few_shot_examples[1] + few_shot_examples[2]
    generate_and_save_results(test_list, three_shot_prompt, 'flan-t5_ddi_three_res.txt', model, tokenizer)

    # Five-shot
    five_shot_prompt = prompt_base + "\n" + few_shot_examples[0] + few_shot_examples[1] + few_shot_examples[2] + \
                       few_shot_examples[3] + few_shot_examples[4]
    generate_and_save_results(test_list, five_shot_prompt, 'flan-t5_ddi_five_res.txt', model, tokenizer)
