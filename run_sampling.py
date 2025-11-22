import os
import json
import pandas as pd
import yaml
import time
from pathlib import Path

from sampler.chat_completion_sampler import ChatCompletionSampler, DivFirstSampler
from sampler.vllm_sampler import VLLMSampler

class Args:
    dataset_name = "gsm8k"
    split = "test"
    k_list = [1,4,8,32]
    subset_num = None
    step_by_step_prompt = True
    n_threads = 1
args = Args()

job_dir = Path(f"runs/default/{args.dataset_name}.qwen-1.5b-inst")
try:
    job_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{job_dir}' and its parent directories created successfully.")
except OSError as e:
    print(f"Error creating directory: {e}")
    
prompt_csv_path = f'{job_dir}/{args.split}.prompts.csv'
sampler_config_dir = f'{job_dir}/distilled-50.direct/sample_2'

with open(f"{sampler_config_dir}/sampler_config.yaml", "r") as f:
    sampler_config = yaml.safe_load(f)
sampler_config

prompt_df = pd.read_csv(prompt_csv_path)
print(prompt_df['prompt'][0])
with open(f"{sampler_config_dir}/sampler_config.yaml", "r") as f:
    sampler_config = yaml.safe_load(f)
sampler_config

# Extract tokenizer config and sampler config
tokenizer_config = sampler_config.get("tokenizer", {})
sampler_config_section = sampler_config.get("sampler", {})

# Dynamically load the sampler class
sampler_class_name = sampler_config_section.get("class", "ChatCompletionSampler")
sampler_classes = {
    "ChatCompletionSampler": ChatCompletionSampler,
    "DivFirstSampler": DivFirstSampler,
    "VLLMSampler": VLLMSampler,
}
SamplerClass = VLLMSampler

# Remove keys that are not arguments to SamplerClass.__init__
init_args = {
    # "api_key_name": "VLLM_TOKEN",
    # "base_url": f"http://localhost:{port}/v1",
}
for k, v in sampler_config_section.items():
    if k == "class":
        continue
    # Renaming config keys to match the argument names where needed
    if k == "model_name":
        # init_args["model"] = v
        init_args["model_name_or_path"] = v
    # elif k == "api_key_name": 
    #     continue
    else:
        init_args[k] = v
    
print(init_args)

# Create the sampler
sampler = SamplerClass(**init_args)


prompts = prompt_df['prompt'].apply(lambda x: x + "<think>\n")
print(prompts[0])


response = sampler.complete(prompts, 1)

generations = []
for res, (_, row) in zip(response, prompt_df.iterrows()):
    for out in res.choices:
        generations.append((row['question_id'], row['prompt_id'], out.response_text, None, row['answer'], sampler_config))


gen_df = pd.DataFrame(data=generations, columns=['question_id', 'prompt_id', 'response', 'pred_answer', 'gt_answer', 'sampler_config'])
gen_df

t = time.localtime()
generation_csv_name = f'generations.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.csv'
gen_df.to_csv(f"{sampler_config_dir}/{generation_csv_name}", index=False)





