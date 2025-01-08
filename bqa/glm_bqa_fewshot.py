import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# Define few-shot examples for different configurations
FEW_SHOT_EXAMPLES = {
    "0shot": "",
    "1shot": "Context: This is an example context. Question: Is this true? Yes.\n",
    "2shot": (
        "Context: This is an example context. Question: Is this true? Yes.\n"
        "Context: Another context for a different question. Question: Is this correct? No.\n"
    ),
    "3shot": (
        "Context: This is an example context. Question: Is this true? Yes.\n"
        "Context: Another context for a different question. Question: Is this correct? No.\n"
        "Context: This context relates to a third question. Question: Is this valid? Yes.\n"
    ),
    "5shot": (
        "Context: This is an example context. Question: Is this true? Yes.\n"
        "Context: Another context for a different question. Question: Is this correct? No.\n"
        "Context: This context relates to a third question. Question: Is this valid? Yes.\n"
        "Context: This is yet another context. Question: Is this relevant? No.\n"
        "Context: Final example context. Question: Is this acceptable? Yes.\n"
    )
}

prompt = '''Please answer the following QUESTION with yes or no according to the CONTEXT: '''


def load_test_dataset(file_path):
    data1 = []
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                question = item["questions"]
                context = item["context"]
                data1.append(question + ' ' + context + ' ')
    return data1


if __name__ == '__main__':
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'

    # Load the test dataset from the local file
    test_file_path = '/data/aim_nuist/aim_zhujj/bqa/Task7B_yesno_test.json'
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

    # Iterate through each few-shot configuration
    for shot in ["0shot", "1shot", "2shot", "3shot", "5shot"]:
        full_prompt = prompt + FEW_SHOT_EXAMPLES[shot]  # Combine base prompt with few-shot examples

        for test in test_list:
            try:
                # Adjusted input tokenization
                inputs = tokenizer(
                    full_prompt + "\n" + test,
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

                    # Combine the generated labels with the result format
                    result = f"{generated_labels}\n"

                    # Write results to respective output file
                    output_file = f'glm_bqa_fewshot_res_{shot}.txt'
                    with open(output_file, 'a', encoding='utf-8') as f:
                        print(result)
                        f.write(result)
            except Exception as e:
                print(f"Error processing test case: {test}\nError: {e}")
