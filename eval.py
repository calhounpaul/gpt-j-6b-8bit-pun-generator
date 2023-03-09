import os

hfloc = "pcalhoun/gpt-j-6b-8bit-pun-generator"
filename = "gptj8bit.pt"

from huggingface_hub import hf_hub_download

#the next few hundred lines are because HF stull doesn't support directly loading models trained in 8bit
DOWNLOAD_LOC = hf_hub_download(repo_id=hfloc, filename=filename)

import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import random, json, datetime

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    GPTNeoForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    GPT2Tokenizer
)

from transformers import GPTNeoForSequenceClassification
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import os
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logging

import transformers
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from tqdm.auto import tqdm

class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias
 
    def forward(self, input):
        output = torch.clone(DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias))
        if self.adapter:
            output += self.adapter(input)
        return output
 
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
 
 
class DequantizeAndLinear(torch.autograd.Function): 
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)
 
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias
 
 
class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
 
    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output 
 
    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"
 
 
def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)
 
    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)

 
def convert_to_int8(model):
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr( 
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )

class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)

        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)
        

class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J

config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

def add_adapters(model, adapter_dim=16):
    assert adapter_dim > 0
    for module in model.modules():
        if isinstance(module, FrozenBNBLinear):
            module.adapter = nn.Sequential(
                nn.Linear(module.in_features, adapter_dim, bias=False),
                nn.Linear(adapter_dim, module.out_features, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)
        elif isinstance(module, FrozenBNBEmbedding):
            module.adapter = nn.Sequential(
                nn.Embedding(module.num_embeddings, adapter_dim),
                nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)

from transformers import StoppingCriteria, StoppingCriteriaList
#modified from the EndOfFunctionCriteria class I found somewhere around here:
# https://huggingface.co/transformers/v4.6.0/_modules/transformers/generation_stopping_criteria.html
class EndOfXCriteria(StoppingCriteria):
    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
    def __call__(self, input_ids, scores, **kwargs):
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)

from datasets import load_dataset
import os
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
import json

gpt = torch.load(DOWNLOAD_LOC,  map_location=torch.device('cuda'))

eval_dir = "eval_logs"

#Punbot was partly trained as an experiment to see if <|extratoken_X|>s could be used for anything.
#I couldn't find much about them on google. They're just a quirk of the way the vocabulary was fit
# to the TPUs, but I thought fine tuning with tokens that weren't in the training set might help
# explain the "anomalous token" phenomenon.
#Additional info: https://paulcalhoun.substack.com/p/this-is-the-moment-ive-been-training

tk = ["<|extratoken_{}|>".format(i) for i in range(10,145)]

PUN_BEGIN = "".join(tk[0:2]) #<|extratoken_10|><|extratoken_11|>
PUN_END =  "".join(tk[2:4]) #<|extratoken_12|><|extratoken_13|>
HOMOPHONES_START =  "".join(tk[4:6])  #<|extratoken_14|><|extratoken_15|>
HOMOPHONES_SEP =  "".join(tk[6:8])  #<|extratoken_16|><|extratoken_17|>
HOMOPHONES_END =  "".join(tk[8:10]) #<|extratoken_18|><|extratoken_19|>
PHONEMES_BEGIN =  "".join(tk[10:12]) #<|extratoken_20|><|extratoken_21|>
PHONEMES_END =  "".join(tk[12:14]) #<|extratoken_22|><|extratoken_23|>
GRAPHEMES_BEGIN =  "".join(tk[14:16]) #<|extratoken_24|><|extratoken_25|>
GRAPHEMES_END =  "".join(tk[16:18]) #<|extratoken_26|><|extratoken_27|>
EXPLANATION_BEGIN =  "".join(tk[18:20]) #<|extratoken_28|><|extratoken_29|>
EXPLANATION_END =  "".join(tk[20:22]) #<|extratoken_30|><|extratoken_31|>
KEYWORDS_BEGIN =  "".join(tk[22:24]) #<|extratoken_32|><|extratoken_33|>
KEYWORDS_END =  "".join(tk[24:26]) #<|extratoken_34|><|extratoken_35|>

GENERATE_KEYWORDS_THEN_EXPLANATION_OF_PUN =  "".join(tk[30:32]) #<|extratoken_40|><|extratoken_41|>
GENERATE_PUN_FROM_EXPLANATION_THEN_KEYWORDS =  "".join(tk[32:34]) #<|extratoken_42|><|extratoken_43|>
#GENERATE_EXPLANATION_THEN_KEYWORDS_FROM_PUN =  "".join(tk[34:36]) #<|extratoken_44|><|extratoken_45|>
GENERATE_PUN_THEN_EXPLANATION_FROM_KEYWORDS =  "".join(tk[36:38]) #<|extratoken_46|><|extratoken_47|>
GENERATE_HOMOPHONE_LIST_FROM_WORD =  "".join(tk[38:40]) #<|extratoken_48|><|extratoken_49|>

TASK_START = "".join(tk[50:52]) #<|extratoken_60|><|extratoken_61|>
TASK_END = "".join(tk[52:54]) #<|extratoken_62|><|extratoken_63|>
GENERATE_EXPLANATION_THEN_PUN_FROM_KEYWORDS =  "".join(tk[54:56]) #<|extratoken_64|><|extratoken_65|>

preprompt = TASK_START + GENERATE_PUN_THEN_EXPLANATION_FROM_KEYWORDS + PUN_BEGIN + GRAPHEMES_BEGIN

def generate(prompt = "",preprompt=preprompt,stop_tokens=tk[40:],max_length=256, top_k=50, top_p=0.98):
    full_prompt=preprompt + prompt
    with torch.no_grad():
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        tkn_len = len(tokenizer.tokenize(full_prompt))
        outputs = gpt.generate(
            inputs["input_ids"],
            stopping_criteria=StoppingCriteriaList(
                [EndOfXCriteria(
                    tkn_len,
                    tk,
                    tokenizer
                )]),
            max_length=max_length,
            do_sample=True, top_k=top_k, top_p=top_p, pad_token_id=50256)
        text = tokenizer.decode(outputs[0])
    fixed_text = text[len(full_prompt) - len(prompt):]
    for token in tk:
        fixed_text = fixed_text.replace(token,"")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    with open(eval_dir + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "".join([c.lower() if c.isalnum() else "_" for c in str(fixed_text)])[:20] + ".txt", "w") as f:
        f.write(text)
    return fixed_text

#create single log file for this run with datetime stamp in filename, and append all jokes to it as they are generated
log_file =  "run_logs/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_log.txt"

os.makedirs("run_logs",exist_ok=True)
with open(log_file, "w") as f:
    f.write("")

for n in range(99):
    pun = generate(prompt="")
    for token in tk:
        pun = pun.replace(token,"\n").replace("\n\n","\n").strip()
    print(pun)
    print("\n\n")
    with open(log_file, "a") as f:
        f.write(pun + "\n\n")
