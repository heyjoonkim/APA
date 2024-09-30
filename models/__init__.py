
import logging
from time import time
from typing import List, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from vllm import LLM

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def _check_model_type(model_name:str) -> str:
    if 'Llama' in model_name:
        model_type = 'causal'
    elif 'opt' in model_name:
        model_type = 'causal'
    elif 'mistral' in model_name:
        model_type = 'causal'
    elif 'vicuna' in model_name:
        model_type = 'causal'
    elif 'gemma' in model_name:
        model_type = 'causal'
    elif 'deberta' in model_name:
        model_type = 'cls'
    else:
        raise NotImplementedError(f'Model {model_name} not implemented.')
    return model_type


## LOAD AUTO-MODEL ##
def _load_model(model_name:str, cache_path:str, offload_path:str=None, dtype:float=torch.float16, quantization_config:BitsAndBytesConfig=None) -> AutoModelForCausalLM:
    model_type = _check_model_type(model_name)
    
    device_map = 'auto' if quantization_config is None else None
    
    if model_type == 'causal':
        model = AutoModelForCausalLM.from_pretrained(
                                                model_name,
                                                # torch_dtype=dtype,
                                                # low_cpu_mem_usage=True,
                                                device_map=device_map,
                                                offload_folder=offload_path,
                                                cache_dir=cache_path,
                                                quantization_config=quantization_config,
                                            ).eval()
        # model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False)
    elif model_type == 'cls':
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to('cuda').eval()
    else:
        raise NotImplementedError(f'Model {model_name} not implemented.')
    
    return model


## LOAD TOKENIZER ##
def _load_tokenizer(model_name:str) -> AutoTokenizer:
    model_type = _check_model_type(model_name=model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_type == 'causal':
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



# resize tokenizer and embedding
# * Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
# from : https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L65
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
) -> None:
    
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    
    if model is not None:
        print('Resize model token embeddings.')
        model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0 and model is not None:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


## LOAD MODEL AND TOKENIZER ##
def load_model_and_tokenizer(model_name:str=None, cache_path:str=None, offload_path:str=None, load_model:bool=True, quantization_config:BitsAndBytesConfig=None) -> List:
    model = _load_model(model_name=model_name, offload_path=offload_path, cache_path=cache_path, quantization_config=quantization_config) if load_model else None
    tokenizer = _load_tokenizer(model_name)

      
    # set special tokens
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # change vocab size for model and tokenizer
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    return model, tokenizer


## LOAD MODEL AND TOKENIZER via VLLM ##
def _load_vllm_model_and_tokenizer(model_name:str=None, trained_path:str=None, cache_path:str=None, tensor_parallel_size:int=0, stage_index:int=0, load_model:bool=True) -> List:
    
    if load_model:
        if stage_index == 0:
            model = LLM(model=model_name, download_dir=cache_path, tensor_parallel_size=tensor_parallel_size)
        else:
            if 'mistral' in model_name:
                model = LLM(trained_path, max_model_len=25000)
            else:
                model = LLM(trained_path)
    else:
        model = None
    tokenizer = _load_tokenizer(model_name)
    return model, tokenizer



def load_vllm_model_and_tokenizer(args, stage_index, previous_model_checkpoint_path=None, load_model=True):
    logger.info('Start loading model and tokenizer...')
    start_time = time()
    if stage_index == 0:
        model, tokenizer = _load_vllm_model_and_tokenizer(model_name=args.model.name, cache_path=args.model.cache_path, tensor_parallel_size=args.model.tensor_parallel_size, stage_index=stage_index, load_model=load_model)
    else:
        logger.info(f'Load model from the last stage : {stage_index-1}')
        model, tokenizer = _load_vllm_model_and_tokenizer(model_name=args.model.name, trained_path=previous_model_checkpoint_path, cache_path=None, tensor_parallel_size=args.model.tensor_parallel_size, stage_index=stage_index, load_model=load_model)

    end_time = time()
    logger.info(f'Done loading. Total loading time : {end_time - start_time} sec.')
    return model, tokenizer