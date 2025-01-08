import os
import torch
from peft import PeftModel
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


# Main processing and evaluation function
def evaluate_model(test_folder, label_folder, model_path, log_file_path):
    # Open log file to write outputs
    with open(log_file_path, 'w') as log_file:
        log_file.write("Processing started...\n")

        # Load test and label data
        test_data = load_test_dataset(test_folder)
        label_data = load_labels_dataset(label_folder)

        # Load and clean the label options for the prompt
        label_options = load_and_clean_label_options(label_folder)
        label_options_str = ", ".join(label_options)  # Create a string of label options

        # Define the new prompt to ask if the label is true or false based on the options
        prompt = f"""
1+2=   """

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).eval()

        all_true_labels = []
        all_predicted_labels = []

        # Iterate through each test file
        for file_name, test in test_data.items():
            if file_name not in label_data:
                log_file.write(f"Label file not found for {file_name}, skipping.\n")
                continue

            # Load corresponding true label
            true_label = label_data[file_name]

            try:
                # Tokenize input with the prompt
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2500
                ).to('cuda')

                # Generation settings
                gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

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

                    # Log output for each file
                    log_file.write(f"Processed {file_name}: True Label: {true_label}, Predicted Label: {predicted_label}\n")

                    # Calculate accuracy incrementally after each prediction
                    if all_true_labels and all_predicted_labels:
                        accuracy = accuracy_score(all_true_labels, all_predicted_labels)
                        log_file.write(f"Current Accuracy (Match Rate) after {file_name}: {accuracy:.4f}\n")

            except Exception as e:
                log_file.write(f"Error processing test case: {test}\nError: {e}\n")

        # Final accuracy after all files
        if all_true_labels and all_predicted_labels:
            final_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
            log_file.write(f"Final Overall Accuracy (Match Rate): {final_accuracy:.4f}\n")
            return final_accuracy
        else:
            log_file.write("No valid predictions to evaluate.\n")
            return 0


if __name__ == '__main__':
    # Paths to the test and label folders, model, and LoRA
    test_folder = './test'
    label_folder = './labels'
    model_path = "/data/aim_nuist/aim_zhujj/llama3"
    log_file_path = './evaluation_log.txt'  # Output log file

    # Evaluate the model and print incremental Accuracy (Match Rate) to the log file
    accuracy = evaluate_model(test_folder, label_folder, model_path, log_file_path)
    print(f"Final Accuracy (Match Rate): {accuracy:.4f}")
