import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score

# Function to load and clean label options from 'labels' folder
def load_and_clean_label_options(label_folder):
    labels_set = set()  # Use a set to ensure no duplicates
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    for file in label_files:
        file_path = os.path.join(label_folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            label = f.read().strip().lower()  # Convert to lowercase for comparison
            if label not in ['NULL', 'sustaining proliferative signaling', 'enabling replicative immortality', 'resisting cell death', 'inducing angiogenesis']:  # Exclude 'null' or empty strings
                labels_set.add(label)

    return list(labels_set)  # Convert set back to list


# Load test data from the 'test' folder
def load_test_dataset(test_folder):
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.txt')]
    data = {}
    for file in test_files:
        file_path = os.path.join(test_folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            data[file] = text
    return data


# Load labels data from the 'labels' folder
def load_labels_dataset(label_folder):
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    labels = {}
    for file in label_files:
        file_path = os.path.join(label_folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            label = f.read().strip().lower()  # Convert to lowercase for comparison
            # Set label to 'false' if it's 'null' or empty, otherwise 'true'
            if label in ['NULL', 'sustaining proliferative signaling', 'enabling replicative immortality', 'resisting cell death', 'inducing angiogenesis']:
                labels[file] = 'false'
            else:
                labels[file] = 'true'
    return labels


# Few-shot examples to include in the prompt
few_shot_examples = [
    "Text: This indicates sustained cell growth.\ttrue\n",
    "Text: There is no observed effect.\tfalse\n",
    "Text: The tumor cells avoided immune detection.\ttrue\n",
    "Text: No significant changes in the outcome.\tfalse\n",
    "Text: Cellular energy metabolism is altered.\ttrue\n"
]


# Main processing and evaluation function
def evaluate_model(test_folder, label_folder, model_path, shot_level=0):
    # Load test and label data
    test_data = load_test_dataset(test_folder)
    label_data = load_labels_dataset(label_folder)

    # Load and clean the label options for the prompt
    label_options = load_and_clean_label_options(label_folder)
    label_options_str = ", ".join(label_options)  # Create a string of label options

    # Define the base prompt for zero-shot
    prompt_base = f"""
    Determine the label for the following text. If the label is 'NULL' or 'sustaining proliferative signaling' or 'enabling replicative immortality' or 'resisting cell death' or 'inducing angiogenesis', respond with 'false'. 
    If the label belongs to the following valid options [{label_options_str}], respond with 'true'. Otherwise, respond with 'false'.
    """

    # Augment the prompt for few-shot
    if shot_level > 0:
        few_shot_prompt = "examples：".join(few_shot_examples[:shot_level])  # Add examples based on shot level
        prompt = prompt_base + "\n" + few_shot_prompt
    else:
        prompt = prompt_base

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()

    all_true_labels = []
    all_predicted_labels = []

    # Open file for saving results
    output_file = f'glm_results_shot{shot_level}.txt'
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Iterate through each test file
        for file_name, test in test_data.items():
            if file_name not in label_data:
                out_f.write(f"Label file not found for {file_name}, skipping.\n")
                continue

            # Load corresponding true label
            true_label = label_data[file_name]

            try:
                # Tokenize input with the prompt
                inputs = tokenizer(
                    prompt + "\n" + test,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2500
                ).to('cuda')

                # Generation settings
                gen_kwargs = {"max_length": 50, "do_sample": True, "top_k": 1}

                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]  # Remove input part from output

                    # Decode the generated labels
                    generated_label = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

                    # Determine if the generated label is in the label set (true/false)
                    predicted_label = generated_label

                    # Collect true and predicted labels
                    all_true_labels.append(true_label)
                    all_predicted_labels.append(predicted_label)

                    # Save the processed result to file
                    out_f.write(f"Processed {file_name}: True Label: {true_label}, Predicted Label: {predicted_label}\n")

            except Exception as e:
                out_f.write(f"Error processing test case: {test}\nError: {e}\n")

        # Calculate accuracy
        if all_true_labels and all_predicted_labels:
            accuracy = accuracy_score(all_true_labels, all_predicted_labels)
            out_f.write(f"Overall Accuracy (Match Rate): {accuracy}\n")
        else:
            out_f.write("No valid predictions to evaluate.\n")


if __name__ == '__main__':
    # Paths to the test and label folders, model
    test_folder = './test'
    label_folder = './labels'
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'

    # Evaluate the model for different shot levels
    for shot_level in [0, 1, 2, 3, 5]:
        print(f"Evaluating with {shot_level}-shot setting...")
        evaluate_model(test_folder, label_folder, model_path, shot_level=shot_level)
        print(f"Results saved for {shot_level}-shot evaluation.")
