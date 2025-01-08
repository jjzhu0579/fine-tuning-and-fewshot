import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import accuracy_score

# Few-shot examples to include in the prompt
FEW_SHOT_EXAMPLES = [
    ("This text is related to tumor promotion", "true"),
    ("This text is about resisting cell death", "true"),
    ("This text discusses angiogenesis", "true"),
    ("This text is irrelevant to any of the provided categories", "false"),
    ("This text belongs to enabling replicative immortality", "true"),
]


# Function to load and clean label options from 'labels' folder
def load_and_clean_label_options(label_folder):
    labels_set = set()  # Use a set to ensure no duplicates
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    for file in label_files:
        file_path = os.path.join(label_folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            label = f.read().strip().lower()  # Convert to lowercase for comparison
            if label not in ['NULL', 'sustaining proliferative signaling', 'enabling replicative immortality',
                             'resisting cell death', 'inducing angiogenesis']:  # Exclude specific labels
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
            if label in ['NULL', 'sustaining proliferative signaling', 'enabling replicative immortality',
                         'resisting cell death', 'inducing angiogenesis']:
                labels[file] = 'false'
            else:
                labels[file] = 'true'
    return labels


# Create Few-shot prompt with examples
def create_few_shot_prompt(test_text, label_options, num_shots):
    # Start with few-shot examples (limited by num_shots)
    prompt = "Few-shot examples:\n"
    for example_text, label in FEW_SHOT_EXAMPLES[:num_shots]:
        prompt += f"Text: {example_text}\nLabel: {label}\n\n"

    # Add the test text and task prompt
    label_options_str = ", ".join(label_options)
    prompt += f"Now determine the label for the following text. If the label belongs to [{label_options_str}], respond with 'true'. Otherwise, respond with 'false'.\n"
    prompt += f"Text: {test_text}\nLabel: "

    return prompt


# Main processing and evaluation function
def evaluate_model(test_folder, label_folder, model_path, num_shots, output_file):
    # Load test and label data
    test_data = load_test_dataset(test_folder)
    label_data = load_labels_dataset(label_folder)

    # Load and clean the label options for the prompt
    label_options = load_and_clean_label_options(label_folder)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                       trust_remote_code=True).eval()

    all_true_labels = []
    all_predicted_labels = []

    # Open file for saving results
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Iterate through each test file
        for file_name, test in test_data.items():
            if file_name not in label_data:
                out_f.write(f"Label file not found for {file_name}, skipping.\n")
                continue

            # Load corresponding true label
            true_label = label_data[file_name]

            try:
                # Create few-shot prompt with the test text and num_shots examples
                input_text = create_few_shot_prompt(test, label_options, num_shots)

                # Generation settings
                gen_kwargs = {"max_length": 50, "do_sample": True, "top_k": 1}

                with torch.no_grad():
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                    outputs = model.generate(input_ids)
                    predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

                    # Collect true and predicted labels
                    all_true_labels.append(true_label)
                    all_predicted_labels.append(predicted_label)

                    # Save the processed result to file
                    out_f.write(
                        f"Processed {file_name}: True Label: {true_label}, Predicted Label: {predicted_label}\n")

            except Exception as e:
                out_f.write(f"Error processing test case: {test}\nError: {e}\n")

        # Calculate accuracy
        if all_true_labels and all_predicted_labels:
            accuracy = accuracy_score(all_true_labels, all_predicted_labels)
            out_f.write(f"Overall Accuracy (Match Rate) for {num_shots}-shot: {accuracy}\n")
        else:
            out_f.write(f"No valid predictions to evaluate for {num_shots}-shot.\n")


if __name__ == '__main__':
    # Paths to the test and label folders, model
    test_folder = './test'
    label_folder = './labels'
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5'

    # Define the different Few-shot scenarios (0-shot, 1-shot, 2-shot, 3-shot, 5-shot)
    few_shot_scenarios = [0, 1, 2, 3, 5]

    for num_shots in few_shot_scenarios:
        output_file = f't5_{num_shots}shot_results.txt'
        print(f"Evaluating {num_shots}-shot scenario...")
        evaluate_model(test_folder, label_folder, model_path, num_shots, output_file)
        print(f"Results saved for {num_shots}-shot.")
