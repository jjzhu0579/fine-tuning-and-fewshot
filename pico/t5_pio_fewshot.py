import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
# Define base prompt
base_prompt = """Please answer whether the text is one of the entities of participants, interventions, or outcomes in the following format: Answer1(true or false) Answer2(true or false) Answer3(true or false)\n"""

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


def test_model(test_list, shot, model, tokenizer, model_path):
    # Create prompt with the respective few-shot examples
    prompt = base_prompt + shot_examples.get(shot, "")

    output_file = f't5_pio_pt_res_{shot}shot.txt'  # Save output to a separate file for each shot

    for test in test_list:
        try:
            # Tokenize the input with the respective prompt and test text
            input_text = prompt + "\n" + test

            gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

            with torch.no_grad():
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                outputs = model.generate(input_ids)
                predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Combine original input text and generated labels
                result = f"{test}\t{predicted_label}\n"

                # Write results to respective output file
                with open(output_file, 'a', encoding='utf-8') as f:
                    print(result)
                    f.write(result)
        except Exception as e:
            print(f"Error processing test case: {test}\nError: {e}")


if __name__ == '__main__':
    model_path ="/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5"

    # Load the test dataset from the local file
    test_file_path = 'test1.txt'
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
    # Test model for each shot scenario (0-shot, 1-shot, 2-shot, 3-shot, 5-shot)
    for shot in [0, 1, 2, 3, 5]:
        test_model(test_list, shot, model, tokenizer, model_path)
