from typing import List, Dict

import torch
from tqdm import tqdm
from datasets import Dataset
from vllm import SamplingParams, LLM

from pipeline import clean_string
from pipeline.templates import INFERENCE_TEMPLATES
from pipeline.evaluation import get_evaluation_method


@torch.no_grad()
def explicit_ambiguity_detection(
            model:LLM, 
            dataset:Dataset,
            template_id:int,
            evaluation_method:str, 
            greedy_params:SamplingParams,
            correct_threshold:float=None,
            analysis:bool=True,
            **kwargs) -> List[Dict]:
    
    inference_str_list = list()

    assert len(INFERENCE_TEMPLATES) > template_id, f'Invalid template_id {template_id} : len(INFERENCE_TEMPLATES) = {len(INFERENCE_TEMPLATES)}'
    template = INFERENCE_TEMPLATES[template_id]
    
    inference_str_list = [template.format(sample.get('question')) for sample in dataset]
             
    evalaute = get_evaluation_method(evaluation_method)
    
    #########################
    #                       #
    #    GREEDY DECODING    #
    #                       #
    #########################

    # type(generations) : List
    generations = model.generate(inference_str_list, greedy_params)
    generated_texts = [generation.outputs[0].text for generation in generations]
    # clean string (strip(), ...)
    generated_texts = [clean_string(generated_text) for generated_text in generated_texts]

    for generated_text, data in tqdm(zip(generated_texts, dataset), desc=f'Explicit Ambiguity Detection', total=len(dataset)):

        is_ambiguous = data.get('is_ambiguous')
        answers = data.get('answers') if 'answers' in data.keys() else [data.get('answer')]
        
        # evaluation_dict.keys() : score, is_correct
        evaluation_dict = evalaute(prediction=generated_text, answers=answers, is_ambiguous=is_ambiguous, threshold=correct_threshold, analysis=analysis)

        data['prediction'] = generated_text
        data.update(evaluation_dict)

    assert 'is_correct' in dataset[0].keys(), f'is_correct not in dataset[0].keys() : {dataset[0].keys()}'

    return dataset

