import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# Define base prompt
base_prompt = """Please answer whether the text is one of the entities of participants, interventions, or outcomes in the following format: Answer1 Answer2 Answer3 (true or false)\n"""

# Define few-shot examples for different shot scenarios
shot_examples = {
    1: """Text: "The patient received a new medication."
Answer: true false false\n""",

    2: """Text: "The patient received a new medication."
Answer: true false false
Text: "Surgery was performed."
Answer: false true false\n""",

    3: """Text: "The patient received a new medication."
Answer: true false false
Text: "Surgery was performed."
Answer: false true false
Text: "The study involved a control group."
Answer: false true false\n""",

    5: """Text: "The patient received a new medication."
Answer: true false false
Text: "Surgery was performed."
Answer: false true false
Text: "The study involved a control group."
Answer: false true false
Text: "The treatment plan included chemotherapy."
Answer: true false false
Text: "Patients were randomly assigned to groups."
Answer: true false true\n"""
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
                    data.append(line)
                else:
                    skipped_lines += 1
        print(f"Skipped {skipped_lines} empty lines.")
        print(f"Loaded {len(data)} valid test samples.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return data


def extract_first_three_labels(generated_text):
    # Find all occurrences of 'true' or 'false' in the generated text
    labels = re.findall(r'\b(true|false)\b', generated_text)
    # Return the first three labels, or less if there are fewer
    return ' '.join(labels[:3])


def test_model(test_list, shot, model, tokenizer, model_path):
    # Create prompt with the respective few-shot examples
    prompt = base_prompt + shot_examples.get(shot, "")

    output_file = f'glm_pio_pt_res_{shot}shot.txt'  # Save output to a separate file for each shot

    for test in test_list:
        try:
            # Tokenize the input with the respective prompt and test text
            inputs = tokenizer(
                prompt + "\n" + test,
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

                # Extract the first three labels (true/false)
                extracted_labels = extract_first_three_labels(generated_labels)

                # Combine original input text and the first three labels
                result = f"{test}\t{extracted_labels}\n"

                # Write results to respective output file
                with open(output_file, 'a', encoding='utf-8') as f:
                    print(result)
                    f.write(result)
        except Exception as e:
            print(f"Error processing test case: {test}\nError: {e}")


if __name__ == '__main__':
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'

    # Load the test dataset from the local file
    test_file_path = 'test1.txt'
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

    # Test model for each shot scenario (0-shot, 1-shot, 2-shot, 3-shot, 5-shot)
    for shot in [0, 1, 2, 3, 5]:
        test_model(test_list, shot, model, tokenizer, model_path)
