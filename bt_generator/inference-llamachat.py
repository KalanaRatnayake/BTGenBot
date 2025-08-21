from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch, accelerate
import sys
import os
from file_save import save_output_to_file

# Load Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

# List of test files from tasks folder
test_file_list = [
    "original.txt",
    "generative_1.txt",
    "generative_2.txt",
    "generative_3.txt",
    "generative_4.txt",
    "generative_5.txt",
]

if hf_token:
    print("Hugging Face token loaded successfully from environment variable.")
else:
    print("HF_TOKEN environment variable is not set.")

# Models path
model_id = 'meta-llama/Llama-2-7b-chat-hf'

# Adapters path
adapter_id = 'AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter'

# Load quantization configuration
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path = model_id,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path = model_id,
    quantization_config = quantization_config,
    torch_dtype = torch.float16,
    device_map = "auto",
    trust_remote_code = True,
    token=hf_token,
)

# Define the context for the task
context = "<<SYS>> You will be provided a summary of a task performed by a behavior tree, and your objective is to express this behavior tree in XML format.\n <</SYS>>"

# One-shot example
example_task = """The behavior tree represents a robot's navigation system with arm activity. The robot must visit the location "Station A", then follow the aruco with ID=7. The only available actions that must be used in the behavior tree are: "MoveTo", "FollowAruco"."""
example_output = """
<root main_tree_to_execute = "MainTree" >
    <BehaviorTree ID="MainTree">
        <Sequence>
            <MoveTo location="Station A"/>
            <FollowAruco id="7"/>
        </Sequence>
    </BehaviorTree>
</root>
"""

## load base model
base_model.eval()

# Load fine-tuned model
finetuned_model = PeftModel.from_pretrained(base_model, adapter_id, token=hf_token)
finetuned_model = finetuned_model.merge_and_unload()
finetuned_model.eval()

for name in test_file_list:
    print(f"\nRunning inference on task file: {name}")
    task_filename = f"tasks/{name}"

    # load task from the text file
    with open(task_filename, "r") as file:
        task = file.read().strip()

    # zero-shot prompt
    zero_eval_prompt = context + "[INST]" + task + "[/INST]"
    zero_model_input = tokenizer(zero_eval_prompt, return_tensors="pt").to("cuda")

    # one-shot prompt
    one_eval_prompt = context + "[INST]" + example_task + "[/INST]" + example_output + "[INST]" + task + "[/INST]"
    one_model_input = tokenizer(one_eval_prompt, return_tensors="pt").to("cuda")

    ## print task
    print("Task:")
    print(task)

    print("\n zero-shot prompt:")
    print(zero_eval_prompt)

    print("\n one-shot prompt:")
    print(one_eval_prompt)

    for it in range(1, 11): # 10 iterations
        print(f"\nIteration {it} on llamachat:")

        # Evaluate zero-shot with base model
        with torch.no_grad():
            result = tokenizer.decode(base_model.generate(**zero_model_input, max_new_tokens=1000)[0], skip_special_tokens=True)
            print("\nZero-shot base model result:")
            print(result)
            save_output_to_file("llamachat-base", "zero", task_filename, it, result)

        ## Evaluate one-shot with base model
        with torch.no_grad():
            result = tokenizer.decode(base_model.generate(**one_model_input, max_new_tokens=1000)[0], skip_special_tokens=True)
            print("\nOne-shot base model result:")
            print(result)
            save_output_to_file("llamachat-base", "one", task_filename, it, result)

        # Evaluate zero-shot with finetuned model
        with torch.no_grad():
            result = tokenizer.decode(finetuned_model.generate(**zero_model_input, max_new_tokens=1000)[0], skip_special_tokens=True)
            print("\nZero-shot finetuned model result:")
            print(result)
            save_output_to_file("llamachat-finetuned", "zero", task_filename, it, result)

        # Evaluate oneshot with finetuned model
        with torch.no_grad():
            result = tokenizer.decode(finetuned_model.generate(**one_model_input, max_new_tokens=1000)[0], skip_special_tokens=True)
            print("\nOne-shot finetuned model result:")
            print(result)
            save_output_to_file("llamachat-finetuned", "one", task_filename, it, result)