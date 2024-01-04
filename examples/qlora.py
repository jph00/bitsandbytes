import torch, time, os, safetensors
from torch import nn

from peft import get_peft_model, LoraConfig, TaskType
from bitsandbytes.nn import Linear4bit

from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig
from transformers.utils import hub, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME
from bitsandbytes.utils import replace_linear

from accelerate import init_empty_weights

mid = "meta-llama/Llama-2-7b-hf"
#mid = 'princeton-nlp/Sheared-LLaMA-1.3B'
#mid = 'Phind/Phind-CodeLlama-34B-v2'

tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)

with init_empty_weights():
    model = LlamaForCausalLM(cfg).eval()
    model = replace_linear(model, Linear4bit)
model.is_loaded_in_4bit = True

idx = hub.cached_file(mid, SAFE_WEIGHTS_INDEX_NAME)
fns,maps = hub.get_checkpoint_shard_files(mid, idx)

def set_param(module, name, value, device=None, dtype=None):
    mn,_,tn = name.rpartition('.')
    subm = module.get_submodule(mn)
    if device: newv = value.to(device)
    if dtype:  newv = value.to(dtype)
    try:
        param = subm.get_parameter(tn)
        newv = type(param)(newv, requires_grad=param.requires_grad)
    except AttributeError: pass  # it's a buffer
    setattr(subm, tn, newv)

for fn in fns:
    sd = safetensors.torch.load_file(fn)
    for n,p in sd.items(): set_param(model, n, p, dtype=torch.bfloat16)

model.cuda()

tgt = [l+"_proj" for l in ["k", 'v', "q", "o", "gate", "up", "down"]]
peft_config = LoraConfig( r=8, lora_alpha=32, target_modules=tgt, bias="none",
   task_type= "CAUSAL_LM", lora_dropout=0.05, inference_mode= False)
model = get_peft_model(model, peft_config)

prompt = "Jeremy Howard is"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(**inputs.to('cuda'), max_length=20, do_sample=True)
#generate_ids = generate_ids[:, len(inputs['input_ids'][0]):]
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

