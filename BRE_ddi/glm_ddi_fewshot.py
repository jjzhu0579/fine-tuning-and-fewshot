import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Few-shot examples to include in the prompt
FEW_SHOT_EXAMPLES = [
    "Text: The patient showed significant improvement.\ttrue\n",
    "Text: The study failed to show any improvement.\tfalse\n",
    "Text: There was no significant change in the condition.\tfalse\n",
    "Text: The treatment reduced inflammation and pain.\ttrue\n",
    "Text: The outcome was inconclusive.\tfalse\n"
]

# Define the main prompt template
base_prompt = """
Ask whether there is a relationship between the two entities of the sentence: true or false
"""

# Function to load test dataset from a file
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


# Function to create prompt with few-shot examples
def create_few_shot_prompt(test_text, num_shots):
    # Start with the few-shot examples (limited by num_shots)
    prompt = "\n".join(FEW_SHOT_EXAMPLES[:num_shots]) + "\n"
    # Add the main prompt and test text
    prompt += base_prompt + "\n" + test_text
    return prompt


# Main function to evaluate the model and store results for different few-shot scenarios
def evaluate_model_for_few_shots(test_list, model, tokenizer, output_dir, num_shots):
    # Create the output file for this few-shot setting
    output_file = os.path.join(output_dir, f'glm_ddi_{num_shots}shot_res.txt')

    # Iterate through test_list
    for test in test_list:
        try:
            # Generate the prompt with the few-shot examples
            prompt = create_few_shot_prompt(test, num_shots)

            # Tokenize the input prompt
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2500
            ).to('cuda')

            # Generation settings
            gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

            with torch.no_grad():
                # Generate model output
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]

                # Decode the generated labels
                generated_labels = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Combine original test text and generated label
                result = f"{test}\t{generated_labels}\n"

                # Save the result to the output file
                with open(output_file, 'a', encoding='utf-8') as f:
                    print(result)
                    f.write(result)
        except Exception as e:
            print(f"Error processing test case: {test}\nError: {e}")


if __name__ == '__main__':
    # Paths
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'
    test_file_path = 'combined_test.txt'
    output_dir = './results'  # Directory to store results

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the test dataset from the local file
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

    # Define the different Few-shot scenarios (0-shot, 1-shot, 2-shot, 3-shot, 5-shot)
    few_shot_scenarios = [0, 1, 2, 3, 5]

    # Evaluate the model for each Few-shot scenario
    for num_shots in few_shot_scenarios:
        print(f"Evaluating {num_shots}-shot scenario...")
        evaluate_model_for_few_shots(test_list, model, tokenizer, output_dir, num_shots)
        print(f"Results saved for {num_shots}-shot.")
