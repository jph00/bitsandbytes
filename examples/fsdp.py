import torch, time, os, safetensors, torch.multiprocessing as mp
from torch import nn

from peft import get_peft_model, LoraConfig, TaskType
from bitsandbytes.nn import Linear4bit

from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig, LlamaConfig, logging
from transformers.utils import hub, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME
from bitsandbytes.utils import replace_linear

from accelerate import init_empty_weights

import functools
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
import torch.distributed as dist

def lambda_policy_fn(m): return hasattr(m, "weight") and m.weight.requires_grad and not next(m.children(), None)

def policy(module, recurse, nonwrapped_numel):
    if recurse: return True
#     return not next(module.children(), None)
    layers = PrefixEncoder,PromptEncoder,PromptEmbedding,LlamaDecoderLayer
    return isinstance(module, layers) or lambda_policy_fn(module)

pg_file = '/tmp/fsdp_file_test'

def main(rank, world_size):
    def pr(*x): print(rank, *x)
    def p0(*x):
        if rank==0: print(rank, *x)

    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=f'file://{pg_file}')
    torch.cuda.set_device(rank)
    mid = "meta-llama/Llama-2-7b-hf"
    #mid = 'princeton-nlp/Sheared-LLaMA-1.3B'
    #mid = 'Phind/Phind-CodeLlama-34B-v2'

    tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)

    torch.set_default_dtype(torch.bfloat16)
    cfg.use_cache = False
    with init_empty_weights():
        model = LlamaForCausalLM(cfg).eval()
        model.model = replace_linear(model.model, Linear4bit)
    model.is_loaded_in_4bit = True

    def set_param(module, name, value, device=None, dtype=None):
        mn,_,tn = name.rpartition('.')
        subm = module.get_submodule(mn)
        oldval = getattr(subm, tn)
        if not dtype: dtype = oldval.dtype
        value = value.to(dtype)
        if device: value = value.to(device)
        try:
            param = subm.get_parameter(tn)
            value = type(param)(value, requires_grad=param.requires_grad)
        except AttributeError: pass  # it's a buffer
        setattr(subm, tn, value)

    if rank==0:
        idx = hub.cached_file(mid, SAFE_WEIGHTS_INDEX_NAME)
        fns,maps = hub.get_checkpoint_shard_files(mid, idx)

        for fn in fns:
            sd = safetensors.torch.load_file(fn)
            for n,p in sd.items():
                with torch.no_grad(): set_param(model, n, p, device='cpu')
            torch.cuda.empty_cache()

    #model.cuda()
    dist.barrier()

    # TODO: Activation checkpointing
    # TODO: Distributed sampler
    tgt = [l+"_proj" for l in ["k", 'v', "q", "o", "gate", "up", "down"]]
    peft_config = LoraConfig( r=8, lora_alpha=32, target_modules=tgt, bias="none",
       task_type= "CAUSAL_LM", lora_dropout=0.05, inference_mode= False)
    model = get_peft_model(model, peft_config)

    def pif(m):
        if rank>=0: m.to_empty(device=torch.device("cuda"), recurse=False)

    model.to(torch.bfloat16)
    p0('f0')
    model = FSDP(model, auto_wrap_policy=policy, device_id=rank, sync_module_states=True, param_init_fn=pif)
    return p0(model)
    p0('f1')

    prompt = "Jeremy Howard is"
    inputs = tokenizer(prompt, return_tensors="pt").to(rank)

    dist.barrier()
    model.eval()

    import time
    p0('pre')
    with torch.no_grad(): res = model(**inputs)
    p0('post')
    #st = time.process_time()
    #p0('CPU Execution time:', time.process_time()-st, 'seconds')
    #p0(res)

    generate_ids = model.generate(**inputs, max_length=20, do_sample=True)
    p0(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    print(rank, torch.cuda.max_memory_allocated()//1_000_000)

if __name__=='__main__':
    logging.set_verbosity_error()
    torch.manual_seed(1337)
    #WORLD_SIZE = torch.cuda.device_count()
    WORLD_SIZE = 2
    try: os.unlink(pg_file)
    except FileNotFoundError: pass
    mp.spawn(main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    #main(0,1)

