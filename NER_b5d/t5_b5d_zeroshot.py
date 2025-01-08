import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")
model = T5ForConditionalGeneration.from_pretrained("/data/aim_nuist/aim_zhujj/xinjian/FLAN-T5")

# Move the model to the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Read the test data
with open("/data/aim_nuist/aim_zhujj/bc5dis/test.txt", "r") as file:
    test_data = file.readlines()


# Define dataset class
class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx].strip()
        encoding = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return encoding


# Create data loader
test_dataset = CustomTestDataset(test_data, tokenizer)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

# Predict labels and output to output.txt
with open("t5_bc5dis_zeroshot_result.txt", "w") as output_file:
    for line in test_data:
        if line.strip():  # Skip empty lines
            # Construct zero-shot input text
            input_text = '''Generate BIO tags for each word in the given paragraph. The BIO format uses the following labels:
• B-<Entity>: Beginning of an entity
• I-<Entity>: Inside of an entity
• O: Outside of an entity
Please extract all chemicals, genes, and diseases mentioned in the paragraph. Provide the output in the format <word> - <BIO tag>, where each word is followed by its corresponding BIO tag.
''' + line.strip()

            # Tokenize input text and move tensors to the same device as the model
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

            # Generate model output
            outputs = model.generate(input_ids)
            predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Write predicted result to output file
            output_file.write(f"{line.strip()}\t{predicted_label}\n")

print("Zero-shot inference completed and results are saved to 't5_bc5dis_zeroshot_result.txt'.")
