
import os   
import logging
from typing import List, Dict

import ray
import torch
import numpy as np
from time import time
from tqdm import tqdm
from scipy.stats import entropy
from datasets import Dataset
from vllm import LLM, SamplingParams

from utils.data import save_pkl, load_pkl
from models import load_model_and_tokenizer
from pipeline import clean_string
from pipeline.templates import DISAMBIGUATION_TEMPLATES, INFERENCE_TEMPLATES, get_ambiguous_answer

logger = logging.getLogger(__name__)



softmax = torch.nn.Softmax(dim=-1)

class ImplicitMeasure:
        
    def __init__(self, **kwargs) -> None:
        
        for k,v in kwargs.items():
            self.__setattr__(k, v)
                        
        assert os.path.isdir(self.base_output_path), f'{self.base_output_path} does not exist.'
        
        self.disambiguation_template = DISAMBIGUATION_TEMPLATES[self.disambiguation_template_id]
        self.generation_template = INFERENCE_TEMPLATES[self.generation_template_id]
        

# INFOGAIN (ORIGINAL)
class Method_0(ImplicitMeasure):
        
    def __init__(self, **kwargs ) -> None:
        
        super().__init__(**kwargs)
        
        self.puncs = ['.', ',', '!', '?', '~', '-', '_']
        
    def clean_string_for_measure(self, text:str) -> str:
        text = text.strip().lower()
        for punc in self.puncs:
            text = text.replace(punc, '')
        return text
    
    ## ENTROPY MEASURE FOR METHOD 1 ##
    # input_ids.shape: (1, token_length) or (token_length, )
    # logits.shape   : (token_length, vocab_size)
    def _get_per_token_entropy(self, logits, input_ids) -> float:
        # shape: (token_length, )
        if len(input_ids) == 1:
            input_ids = input_ids.squeeze(0)
            
        # shape : (token_length, vocab_size)
        softmax_logits = softmax(logits)
            
        # shape : (token_length, )
        # entropy_values = entropy(softmax_logits, axis=-1, base=2)
        entropy_values = entropy(softmax_logits, axis=-1)

        entropy_value = np.mean(entropy_values)
        
        return entropy_value
    
    ##########################################################
    #                                                        #
    #    GENERATE GREEDY DISAMBIGUATION FOR EACH QUESTION    #
    #                                                        #
    ##########################################################
    
    def greedy_generation(self) -> List[str]:
        output_file = os.path.join(self.base_output_path, 'tmp_greedy_generations.pkl')
        
        if not os.path.isfile(output_file):
            disambiguation_str_list = [self.disambiguation_template.format(sample.get('question')) for sample in self.dataset]
            
            logger.info('Generate disambiguation for each question...')
            # type(disambiguations) : List
            disambiguated_outputs = self.model.generate(disambiguation_str_list, self.greedy_params)
            disambiguated_outputs = [disambiguation.outputs[0] for disambiguation in disambiguated_outputs]

            # type(disambiguated_question)  = List[str]
            disambiguated_questions = [clean_string(disambiguated_output.text) for disambiguated_output in disambiguated_outputs]
                        
            save_pkl(path=output_file, data=disambiguated_questions)
        else:
            logger.info(f'Greedy disambiguation already exists. Load generation from : {output_file}')
        
        greedy_disambiguated_questions = load_pkl(output_file)
        return greedy_disambiguated_questions
        
        
    @torch.no_grad()
    def measure_infogain(self):
        for disambiguated_question, data in tqdm(zip(self.greedy_disambiguated_questions, self.dataset), desc=f'Implicit Ambiguity Detection', total=len(self.dataset)):
            
            ## MEASURE ENTROPY FOR ORIGINAL QUESTION ##
            question = data.get('question')

            # clean string for implicit measure
            question = self.clean_string_for_measure(question)

            question_inputs = self.tokenizer(question, padding=False, truncation=False, return_tensors='pt')
            question_inputs = {k:v.to('cuda') for k,v in question_inputs.items()}
            question_ids = question_inputs.get('input_ids').cpu()
            
            with torch.no_grad():
                question_output = self.model(**question_inputs)
                
            # shape : (1, token_length, vocab_size) -> (token_length, vocab_size)
            question_logits = question_output.logits.detach().cpu().squeeze(0)
            # shape : (input_length, )
            # measure entropy for original quesiton
            question_entropy = self._get_per_token_entropy(logits=question_logits, input_ids=question_ids)


            ## MEASURE ENTROPY FOR DISAMBIGUATION ##
            # clean string for implicit measure
            disambiguated_question = self.clean_string_for_measure(disambiguated_question)
            
            disambiguation_inputs = self.tokenizer(disambiguated_question, padding=False, truncation=False, return_tensors='pt')
            disambiguation_inputs = {k:v.to('cuda') for k,v in disambiguation_inputs.items()}
            disambiguation_ids = disambiguation_inputs.get('input_ids').cpu()
            
            with torch.no_grad():
                disambiguation_output = self.model(**disambiguation_inputs)
                
            # shape : (1, token_length, vocab_size) -> (token_length, vocab_size)
            disambiguation_logits = disambiguation_output.logits.detach().cpu().squeeze(0)
            # shape : (input_length, )
            # measure entropy for original quesiton
            disambiguation_entropy = self._get_per_token_entropy(logits=disambiguation_logits, input_ids=disambiguation_ids)
            
            # disambiguated : more certain -> low entropy
            # ambiguous     : less certain -> high entropy
            # score         : disambiguation_entropy - question_entropy
            # score = disambiguation_entropy - question_entropy
            score = question_entropy - disambiguation_entropy
            
                    
            data.update(
                dict(
                    disambiguation=disambiguated_question,
                    score=score.item(),
                )
            )
        return self.dataset
        
    def measure(self):
        
        ## greedy disambiguations
        self.greedy_disambiguated_questions = self.greedy_generation()
        
        ## remove model and tokenizer for more CUDA memory ##
        if self.model is not None:
            logger.info('Remove model for CUDA memory.')
            del self.model
            torch.cuda.empty_cache()
            if ray.is_initialized():
                ray.shutdown()
        else:
            logger.info('Model is None.')
            
        ## Load model and tokenizer for measuring entropy (information gain) ##
        logger.info(f'Load model: {self.model_name}')
        start_time = time()
        self.model, self.tokenizer = load_model_and_tokenizer(model_name=self.model_name, cache_path=self.cache_path)
        end_time = time()
        logger.info(f'Done loading model. Total loading time : {end_time - start_time} sec.')
        
        results = self.measure_infogain()
        return results


def get_inference_method(method_id:int):
    if method_id == 0:
        return Method_0
    else:
        raise NotImplementedError(f'Inference method {method_id} not implemented.')


@torch.no_grad()
def implicit_ambiguity_detection(
            model:LLM, 
            model_name:str, 
            cache_path:str,
            dataset:Dataset,
            method_id:int,
            # GENERATION CONFIGS #
            greedy_params:SamplingParams,
            sampling_params:SamplingParams,
            disambiguation_template_id:int=None,
            generation_template_id:int=None,
            base_output_path:str=None,
            aggregate_method:str=None,
            **kwargs) -> List[Dict]:
    
    INFERENCE_METHOD = get_inference_method(method_id=method_id)

    # INITIALIZE IMPLICIT INFERENCE METHOD
    inference_method = INFERENCE_METHOD(model=model, 
                                        model_name=model_name,
                                        cache_path=cache_path,
                                        dataset=dataset, 
                                        greedy_params=greedy_params,
                                        sampling_params=sampling_params,
                                        disambiguation_template_id=disambiguation_template_id,
                                        generation_template_id=generation_template_id,
                                        base_output_path=base_output_path,
                                        aggregate_method=aggregate_method)

    results = inference_method.measure()
    
    if model is not None:
        del model
        torch.cuda.empty_cache()
        
    return results



def filter_implicit_ambiguity(dataset, threshold:float) -> Dict[str,List]:
       
    ambiguous_results = list()
    unambiguous_results = list()
    for data in tqdm(dataset, desc='Filtering Implicit Ambiguity [Stage {stage_index}]', total=len(dataset)):
        ambiguity_score = data.get('score')

        if ambiguity_score > threshold:
            ## ADD SAMPLE CLASSIFIED AS AMBIGUOUS ##
            data['prediction'] = get_ambiguous_answer()
            ambiguous_results.append(data)
        else:
            unambiguous_results.append(data)

    return dict(
        ambiguous=ambiguous_results,
        unambiguous=unambiguous_results
    )
        
        