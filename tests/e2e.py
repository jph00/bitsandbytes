import os, time, torch, torch.multiprocessing as mp
from os.path import join
from dataclasses import dataclass, field
from typing import List
from functools import partial

from peft.peft_model import PeftModel
from peft import get_peft_model, LoraConfig, TaskType
import torch.distributed as dist

from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

import bitsandbytes as bnb
from bitsandbytes.distributed.fsdp import bnb_fsdp_auto_wrap_policy, parameters_all_consistent
from bitsandbytes.nn import Linear4bit
from bitsandbytes.utils import replace_linear

from models import HierarchicalModel, SimpleModel, MoreComplexModel, LoraDecoder
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from fsdp_utils import fsdp_auto_wrap_policy

pg_file = '/tmp/fsdp_file_test'

def to_text(x):
    x['text'] = 'Context: ' + x['context'] + '\nQuestion: ' + x['question'] + '\nAnswer: ' + x['answer']
    # tokenize here?
    return x

def tok(x):
    x = tokenizer(x['text'], padding='longest')
    x['labels'] = deepcopy(x['input_ids'])
    return x

def main(rank, world_size):
    def p0(x): print(x) if rank==0 else None
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=f'file://{pg_file}')
    torch.cuda.set_device(rank)

    tgt = [l+"_proj" for l in ["k", 'v', "q", "o", "gate", "up", "down"]]
    peft_config = LoraConfig( r=8, lora_alpha=32, target_modules=tgt, bias="none",
        task_type= "CAUSAL_LM", lora_dropout=0.05, inference_mode= False)

    model_name = 'PY007/TinyLlama-1.1B-intermediate-step-480k-1T'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    #replacement = partial(Linear4bit, compute_dtype=torch.bfloat16)
    #model = replace_linear(model, replacement)
    model = get_peft_model(model, peft_config)
    if rank==0: model.print_trainable_parameters()

    wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
    model = FSDP(model, auto_wrap_policy=wrapping_policy,
        device_id=torch.cuda.current_device(), limit_all_gathers=True,
        mixed_precision=None, sync_module_states=False,)
    #p0(model)
    #return

    tokr = AutoTokenizer.from_pretrained(model_name)
    tokr.pad_token = tokr.eos_token
    ds = load_dataset("knowrohit07/know_sql", revision='f33425d13f9e8aab1b46fa945326e9356d6d5726', split="train")
    ds = ds.select(range(0,len(ds),10))
    ds = ds.shuffle(42).map(to_text)
    ds = ds.map(lambda x: {"lens": len(x["text"])}).filter(lambda x:x['lens']<380).sort('lens').remove_columns(['lens'])
    trn = ds.select(range(0, len(ds)-200))
    val = ds.select(range(len(ds)-200, len(ds)))

    model.train()
    p0('done')

if __name__=='__main__':
    torch.manual_seed(1337)
    WORLD_SIZE = torch.cuda.device_count()
    WORLD_SIZE = 2
    try: os.unlink(pg_file)
    except FileNotFoundError: pass
    mp.spawn(main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

