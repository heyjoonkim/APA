import os
import random
import logging
from typing import List, Dict
from utils.logging import logger_init

from vllm import SamplingParams

from pipeline import clean_string
from models import load_vllm_model_and_tokenizer
from utils.data import save_pkl, load_pkl

logger = logging.getLogger(__name__)


def data_selection(base_output_path:str,
                    ABLATION_METHODS:List[int],
                    ambiguous_results:List,
                    unambiguous_results:List,
                    correct_samples:List,
                    implicit_results:List,          # same as incorect_samples
                    model=None,
                    args=None,
                    greedy_params:SamplingParams=None,
                    sampling_params:SamplingParams=None,)->List[Dict]:
        
    logger_init(logger, base_output_path, save_as_file=True)
    
    ABLATION_METHOD_IDS = list()
    
    if 1 in ABLATION_METHODS:
        ABLATION_METHOD_IDS.append(1)
        logger.info('[ANNOTATION METHOD 1] Balance training samples.')
        BALANCED_SAMPLE_SIZE = min(len(ambiguous_results), len(correct_samples))
        if BALANCED_SAMPLE_SIZE < len(correct_samples):
            org_size = len(correct_samples)
            correct_samples = random.sample(correct_samples, BALANCED_SAMPLE_SIZE)
            logger.info(f'Randomly select {BALANCED_SAMPLE_SIZE} samples from CORRECT samples. {org_size} => {BALANCED_SAMPLE_SIZE}')
        elif BALANCED_SAMPLE_SIZE < len(ambiguous_results):
            org_size = len(ambiguous_results)
            sorted_ambiguous_results = sorted(ambiguous_results, key=lambda x: x.get('score'))
            
            ambiguous_results = sorted_ambiguous_results[-BALANCED_SAMPLE_SIZE:]
            assert len(ambiguous_results) == BALANCED_SAMPLE_SIZE
            logger.info(f'Randomly select {BALANCED_SAMPLE_SIZE} samples from IMPLICITLY AMBIGUOUS samples. {org_size} => {BALANCED_SAMPLE_SIZE}')
            selected_ambig = [d for d in ambiguous_results if d.get('is_ambiguous')]
            selected_unambig = [d for d in ambiguous_results if d.get('is_ambiguous') is False]
            ratio = len(selected_ambig) / (len(selected_ambig) + len(selected_unambig))
            logger.info(f'Selected {BALANCED_SAMPLE_SIZE} samples : Ambig ({len(selected_ambig)}), Unambig({len(selected_unambig)}) (Ratio : {round(ratio*100, 2)})')

        assert len(correct_samples) == len(ambiguous_results), f'Balanced samples are not same. Correct: {len(correct_samples)}, Ambiguous: {len(ambiguous_results)}'
    
    if 2 in ABLATION_METHODS:
        ABLATION_METHOD_IDS.append(2)
        logger.info('[ANNOTATION METHOD 2] Generate explanation for the ambiguity.')
        EXPLANATION_TEMPLATE_ID = args.pipeline.explanation.template_id
        ablation_methods_str = '-'.join([str(method_id) for method_id in ABLATION_METHODS])
        output_file = os.path.join(base_output_path, f'tmp_selected_data_method_{ablation_methods_str}_explanation_generations_{EXPLANATION_TEMPLATE_ID}.pkl')
        
        if not os.path.isfile(output_file):
            from pipeline.templates import EXPLANATION_TEMPLATE
            explanation_template = EXPLANATION_TEMPLATE[EXPLANATION_TEMPLATE_ID]
            
            if model is None:
                model, _ = load_vllm_model_and_tokenizer(args, stage_index=0)
                
            inference_str_list = [explanation_template.format(sample.get('question'), sample.get('disambiguation')) for sample in ambiguous_results]
                    
            # type(generations) : List
            generations = model.generate(inference_str_list, greedy_params)
            generated_texts = [generation.outputs[0].text for generation in generations]
            # clean string (strip(), ...)
            generated_texts = [clean_string(generated_text) for generated_text in generated_texts]
            
            for generated_text, ambiguous_result in zip(generated_texts, ambiguous_results):
                ambiguous_result['explanation'] = generated_text
                ambiguous_result['prediction'] = generated_text
            save_pkl(data=ambiguous_results, path=output_file)
        ambiguous_results = load_pkl(path=output_file)
                
                
    if 0 in ABLATION_METHODS:
        ABLATION_METHOD_IDS.append(0)
        ## RE-LABELING PROCESS ##
        for correct_sample in correct_samples:
            is_ambiguous = correct_sample.get('is_ambiguous')
            if not is_ambiguous:
                if 'answers' in correct_sample.keys():
                    # for ambigqa
                    answer = correct_sample.get('answers')[0]
                else:
                    # for ambig_trivia_qa, ambig_wiki_qa
                    answer = correct_sample.get('answer')
                correct_sample['prediction'] = answer
            else:
                # for ambiguous+correct samples, use the generated answers
                continue
                
    ## BUILD FINAL DATASET ##
    selected_train_data = correct_samples + ambiguous_results
    random.shuffle(selected_train_data)
    
    logger.info(f'Selected Training Data ({len(correct_samples) + len(ambiguous_results)}): Correct ({len(correct_samples)}), Ambiguous ({len(ambiguous_results)})')
        
    for ablation_method_id in ABLATION_METHOD_IDS:
        assert ablation_method_id in ABLATION_METHODS
    
    return selected_train_data, ABLATION_METHOD_IDS