from typing import Dict

from transformers import AutoTokenizer
from vllm import SamplingParams

BAD_WORDS = ['Q:', 'A:', 'Question:', 'Answer:', '\n', '\t', '\r']


def get_generation_config(args, tokenizer:AutoTokenizer) -> Dict:
    bad_words = ['Q:', ]
    
    eos_tokens = ['\n', ',', '.']
    
    if tokenizer.__class__.__name__ == 'LlamaTokenizerFast':
        eos_token_id = [tokenizer(token)['input_ids'][-1] for token in eos_tokens]
    else:
        raise NotImplementedError('Tokenzier not implemented.')

    
    eos_token_id += [tokenizer.eos_token_id]
    bad_words_ids = [tokenizer(token)['input_ids'] for token in bad_words]
    
    generation_config = dict(
        eos_token_id=eos_token_id, 
        bad_words_ids=bad_words_ids,
        max_new_tokens = args.generation.max_new_tokens,
        min_new_tokens = 2,
        pad_token_id = tokenizer.eos_token_id
    )

    return generation_config


def get_vllm_param(args, do_sampling:bool=True) -> Dict:
    
    greedy_params = SamplingParams(n=1, temperature=0, stop=BAD_WORDS, max_tokens=args.generation.max_new_tokens)
        
    if do_sampling:
        sampling_params = SamplingParams(
                            n=args.generation.num_generations_per_prompt, 
                            temperature=args.generation.temperature, 
                            stop=BAD_WORDS, 
                            max_tokens=args.generation.max_new_tokens)
    else:
        sampling_params = None
    
    return dict(
        greedy=greedy_params,
        sampling=sampling_params
    )



def clean_string(input_str:str) -> str:
    input_str = input_str.strip()
    FILTER_STRINGS = ['\n', '\t', '\r']
    for filter_string in FILTER_STRINGS:
        if filter_string in input_str:
            input_str = input_str.split(filter_string)[0]

    if input_str == '':
        input_str = '.'

    return input_str
