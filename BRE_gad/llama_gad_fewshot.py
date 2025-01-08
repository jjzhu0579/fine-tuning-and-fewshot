import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Base prompt for zero-shot
prompt_base = """
please select labels from the following range according to the text : true,false
"""

# Few-shot examples for 1-shot, 2-shot, 3-shot, 5-shot
few_shot_examples = [
    "Text: The patient showed significant improvement.\tLabel: true\n",
    "Text: The study failed to show any improvement.\tLabel: false\n",
    "Text: There was no significant change in the condition.\tLabel: false\n",
    "Text: The treatment reduced inflammation and pain.\tLabel: true\n",
    "Text: The outcome was inconclusive.\tLabel: false\n"
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


def generate_and_save_results(test_list, prompt, output_file):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

    # Iterate through test_list
    for test in test_list:
        try:
            # Adjust input tokenization
            inputs = tokenizer(
                prompt + "\n" + test,
                return_tensors="pt",
                truncation=True,
                max_length=2500
            ).to('cuda')

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]

                # Decode the generated labels
                generated_labels = tokenizer.decode(outputs[0], skip_special_tokens=True)
                labels_to_find = ['true', 'false']

                # Find the earliest occurrence of a label in the generated output
                earliest_label = None
                earliest_index = len(generated_labels)

                for label in labels_to_find:
                    index = generated_labels.find(label)
                    if 0 <= index < earliest_index:
                        earliest_index = index
                        earliest_label = label

                if earliest_label:
                    generated_labels = earliest_label

                # Output the result
                print(generated_labels)

                # Combine original text and generated label
                result = f"{test}\t{generated_labels}\n"

                # Save the result to the output file
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(result)
        except Exception as e:
            print(f"Error processing test case: {test}\nError: {e}")


if __name__ == '__main__':
    model_path = "/data/aim_nuist/aim_zhujj/llama3"

    # Load the test dataset
    test_file_path = 'test1.tsv'
    test_list = load_test_dataset(test_file_path)

    # 0-shot
    zero_shot_prompt = prompt_base
    generate_and_save_results(test_list, zero_shot_prompt, 'llama_gad_zero_res.txt')

    # 1-shot
    one_shot_prompt = prompt_base + "\n" + few_shot_examples[0]
    generate_and_save_results(test_list, one_shot_prompt, 'llama_gad_one_res.txt')

    # 2-shot
    two_shot_prompt = prompt_base + "\n" + few_shot_examples[0] + few_shot_examples[1]
    generate_and_save_results(test_list, two_shot_prompt, 'llama_gad_two_res.txt')

    # 3-shot
    three_shot_prompt = prompt_base + "\n" + few_shot_examples[0] + few_shot_examples[1] + few_shot_examples[2]
    generate_and_save_results(test_list, three_shot_prompt, 'llama_gad_three_res.txt')

    # 5-shot
    five_shot_prompt = prompt_base + "\n" + few_shot_examples[0] + few_shot_examples[1] + few_shot_examples[2] + few_shot_examples[3] + few_shot_examples[4]
    generate_and_save_results(test_list, five_shot_prompt, 'llama_gad_five_res.txt')
