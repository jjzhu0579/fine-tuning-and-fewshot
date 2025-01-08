import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define few-shot examples for different configurations
FEW_SHOT_EXAMPLES = {
    "0shot": "",
    "1shot": "There is a relationship between GENE1 and DISEASE1: true\n",
    "2shot": "There is a relationship between GENE1 and DISEASE1: true\nThere is no relationship between GENE2 and DISEASE2: false\n",
    "3shot": "There is a relationship between GENE1 and DISEASE1: true\nThere is no relationship between GENE2 and DISEASE2: false\nThe interaction between GENE3 and DISEASE3 is confirmed: true\n",
    "5shot": (
        "There is a relationship between GENE1 and DISEASE1: true\n"
        "There is no relationship between GENE2 and DISEASE2: false\n"
        "The interaction between GENE3 and DISEASE3 is confirmed: true\n"
        "No known association exists between GENE4 and DISEASE4: false\n"
        "The entities GENE5 and DISEASE5 are unrelated: false\n"
    )
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


if __name__ == '__main__':
    model_path = '/data/aim_nuist/aim_zhujj/xinjian/glm4_lora_ptuning/ZhipuAI/glm-4-9b-chat'


    # Load the test dataset from the local file
    test_file_path = 'chemprot_test.txt'
    test_list = load_test_dataset(test_file_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()


    # Iterate through each few-shot configuration
    for shot in ["0shot", "1shot", "2shot", "3shot", "5shot"]:
        prompt = f"please select labels from the following range according to the text : true,false\n{FEW_SHOT_EXAMPLES[shot]}"

        for test in test_list:
            try:
                # Adjusted input tokenization
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

                    # Combine original text and generated labels
                    result = f"{test}\t{generated_labels}\n"

                    # Write results to respective output file
                    output_file = f'glm_cp_res_{shot}.txt'
                    with open(output_file, 'a', encoding='utf-8') as f:
                        print(result)
                        f.write(result)
            except Exception as e:
                print(f"Error processing test case: {test}\nError: {e}")
