import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
prompt = '''Please answer the following QUESTION with yes or no according to the CONTEXT: '''


def load_test_dataset(file_path):
    data1 = []
    texts = []
    labels = []
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                question = item["questions"]
                context = item["context"]
                data1.append(question+' '+context+' ')
    return data1


if __name__ == '__main__':
    model_path = "/data/aim_nuist/aim_zhujj/llama3"
    lora_path = '/data/aim_nuist/aim_zhujj/llama3/llama_lora_bqa'

    test_file_path = 'Task7B_yesno_test.json'
    test_list = load_test_dataset(test_file_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Set `bos_token_id` to a reasonable value if it's None
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.cls_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    # Iterate through test_list
    for test in test_list:
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": test}
            ]

            # Assuming `apply_chat_template` is a valid method; if not, replace it accordingly
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

            generated_ids = model.generate(
                input_ids=model_inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.5,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,  # or define explicitly if necessary
            )

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            original_text = test
            print(response)
            result = f"{original_text}\t{response}\n"

            with open('lla_lo_bqa_res.txt', 'a', encoding='utf-8') as f:
                print(result)
                f.write(result)
        except Exception as e:
            print(f"Error processing test case: {test}\nError: {e}")
