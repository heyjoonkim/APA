#########################################
#   STAGE 0 PROCESS                     #
#   1. Explicit evaluation on train set #
#   2. Implict ambiguit measure         #
#   3. select ambiguous samples         #
#########################################

import logging
import os
from time import time

import hydra
from omegaconf import DictConfig

from utils.logging import logger_init
from utils.data import save_pkl, load_pkl
from utils import seed_everything
from data.dataclass.utils import get_dataset_class
from models import load_vllm_model_and_tokenizer
from pipeline import get_vllm_param
from pipeline.explicit import explicit_ambiguity_detection
from pipeline.implicit import implicit_ambiguity_detection, filter_implicit_ambiguity
from pipeline.evaluation import analysis
from pipeline.preprocess import make_supervised_data_module

logger = logging.getLogger(__name__)


@hydra.main(config_path='../configs/', config_name='main.yaml', version_base=None)
def main(args: DictConfig) -> None:

    random_seed = args.seed

    seed_everything(random_seed)

    # set path
    model_name = args.model.name
    model_name_for_dir = model_name.replace('/', '-')

    ## tmp path for outputs
    tmp_path = os.path.join(args.path.output, model_name_for_dir, str(random_seed))

    ####################################################
    #                                                  #
    # SET OUTPUT PATHS & FILES FOR EACH PIPELINE STAGE #
    #                                                  #
    ####################################################
    
    ### HYPERPARAMETERS ####
    STAGE_INDEX = 0
    INFERENCE_TEMPLATE_ID = args.pipeline.explicit.template_id
    IMPLICIT_METHOD_ID = args.pipeline.implicit.method_id
    DISAMBIGUATION_TEMPLATE_ID = args.pipeline.implicit.disambiguation_template_id
    EVALUATION_METHOD = args.pipeline.explicit.evaluation_method
    ABLATION_METHODS = args.ablation_methods
    # only if 2 in ABLATION_METHODS
    EXPLANATION_TEMPLATE_ID = args.pipeline.explanation.template_id
    ablation_methods_str = '-'.join([str(method_id) for method_id in ABLATION_METHODS])

    ### BASE PATH FOR OUTPUTS ###
    base_output_path = os.path.join(tmp_path, f'explicit_template_{INFERENCE_TEMPLATE_ID}_implicit_method_{IMPLICIT_METHOD_ID}_template_{DISAMBIGUATION_TEMPLATE_ID}')
   
    if not os.path.isdir(base_output_path):
        os.makedirs(base_output_path, exist_ok=True)
     

    logger.info(f'SET OUTPUT PATHS & FILES FOR @ STAGE {STAGE_INDEX}')
    # stage 1   : EXPLICIT generation directory
    # stage 1-1 : explicit prediction on training set (check whether the model can handle such inputs.)
    explicit_output_file = os.path.join(base_output_path, f'1_explicit_{EVALUATION_METHOD}_results_stage_{STAGE_INDEX}.pkl')

    # stage 2   : IMPLICIT generation directory
    # stage 2-1 : implicit ambiguity prediction on training set (generate disambiguations for initial samples)
    implicit_generation_output_file = os.path.join(base_output_path, f'2_implicit_generation_stage_{STAGE_INDEX}.pkl')
    implicit_generation_tmp_file = os.path.join(base_output_path, f'2_implicit_generation_tmp_stage_{STAGE_INDEX}.pkl')
    # stage 2-2 : selected samples via implicit ambiguity prediction measures (measure infogain)
    filter_threshold = args.pipeline.implicit.threshold
    implicit_output_file = os.path.join(base_output_path, f'2_implicit_threshold_{filter_threshold}_stage_{STAGE_INDEX}.pkl')

    # stage 3   : final selected data directory (samples selected in stage 1 + stage 2)
    # stage 3-1 : final selected data
    selected_data_file = os.path.join(base_output_path, f'3_selected_data_method_{ablation_methods_str}_threshold_{filter_threshold}_stage_{STAGE_INDEX}.pkl')
    if 2 in ABLATION_METHODS:
        selected_data_file = os.path.join(base_output_path, f'3_selected_data_method_{ablation_methods_str}_threshold_{filter_threshold}_explanation_template_{EXPLANATION_TEMPLATE_ID}_stage_{STAGE_INDEX}.pkl')


    ## INIT LOGGER
    logger_level = args.logging_level
    logger_init(logger, base_output_path, logger_level=logger_level, save_as_file=True)


    ############################
    #                          #
    #       LOAD DATASET       #
    #                          #
    ############################
        
    ## LOAD DATASET FOR IN-DOMAIN TRAINING ##
    dataset_path = os.path.join(args.path.data, 'ambigqa')
    dataset_class = get_dataset_class(name='ambigqa', data_path=dataset_path, seed=random_seed)
    dataset_dict = dataset_class.get_dataset()
    train_dataset = dataset_dict.get('train')

    logger.info(f'Load TRAIN set (AmbigQA): {len(train_dataset)} samples.')

    ####################################
    #                                  #
    #   LOAD MODEL FOR FULL PIPELINE   #
    #                                  #
    ####################################

    model = None
    tokenizer = None
    
    ##################################################
    #                                                #
    # GET VLLM GENERATION CONFIGS (Greedy, Sampling) #
    #                                                #
    ##################################################

    params_dict = get_vllm_param(args)
    greedy_params = params_dict.get('greedy')
    sampling_params = params_dict.get('sampling')
    
    ###############################################
    #                                             #
    # STAGE 1: EXPLICIT AMBIGUITY DETECTION STAGE #
    #                                             #
    ###############################################

    if not os.path.isfile(explicit_output_file):
        
        if model is None:
            model, tokenizer = load_vllm_model_and_tokenizer(args, stage_index=STAGE_INDEX)
            
        logger.info(f'EXPLICIT prediction @ stage {STAGE_INDEX}.')
        start_time = time()
        explicit_results = explicit_ambiguity_detection(
                                                        model=model,
                                                        dataset=train_dataset,
                                                        template_id=INFERENCE_TEMPLATE_ID,
                                                        evaluation_method=args.pipeline.explicit.evaluation_method,
                                                        correct_threshold=args.pipeline.explicit.correct_threshold,
                                                        greedy_params=greedy_params,
                                                        analysis=True,
                                                    )
        end_time = time()
        save_pkl(data=explicit_results, path=explicit_output_file)
        logger.info(f'Done. Total time : {end_time - start_time} sec.')
    else:
        logger.info(f'EXPLICIT prediction results already exists @ stage {STAGE_INDEX}. Load from : {explicit_output_file}')

    # score, is_correct, prediction
    explicit_results = load_pkl(path=explicit_output_file)

    # Use correct samples as training set
    correct_samples = [sample for sample in explicit_results if sample.get('is_correct') is True]
    # Use incorrect samples for implicit ambiguity detection stage
    incorrect_samples = [sample for sample in explicit_results if sample.get('is_correct') is False]
    
    logger.info(f'Done explicit prediction. Correct {len(correct_samples)} samples, incorrect {len(incorrect_samples)} samples.')

    analysis(dataset=explicit_results, logger=logger)


    ######################################
    #                                    #
    # IMPLICIT AMBIGUITY DETECTION STAGE #
    #                                    #
    ######################################

    ## GENERATE DISAMBIGUATIONS ##
    if not os.path.isfile(implicit_generation_output_file):
        
        # load model is only disambiguation file does not exist
        if not os.path.isfile(implicit_generation_tmp_file) and model is None:
            model, tokenizer = load_vllm_model_and_tokenizer(args, stage_index=STAGE_INDEX)
            
        logger.info(f'IMPLICIT ambiguity detection @ stage {STAGE_INDEX}.')
        start_time = time()
        
        implicit_results = implicit_ambiguity_detection(model=model,
                                                        model_name=model_name,
                                                        cache_path=args.model.cache_path,
                                                        dataset=incorrect_samples,
                                                        method_id=IMPLICIT_METHOD_ID,
                                                        disambiguation_template_id=DISAMBIGUATION_TEMPLATE_ID,
                                                        generation_template_id=INFERENCE_TEMPLATE_ID,
                                                        sampling_params=sampling_params,
                                                        greedy_params=greedy_params,
                                                        base_output_path=base_output_path,
                                                        aggregate_method=args.pipeline.implicit.aggregate_method,)
        end_time = time()
        save_pkl(data=implicit_results, path=implicit_generation_output_file)
        logger.info(f'Done. Total time : {end_time - start_time} sec.')
    else:
        logger.info(f'IMPLICIT ambiguity detection results already exists @ stage {STAGE_INDEX}.')

    implicit_results = load_pkl(path=implicit_generation_output_file)

    
    ## FILTER OUT VAILD SAMPLES ##
    if not os.path.isfile(implicit_output_file):
        results_dict = filter_implicit_ambiguity(dataset=implicit_results, threshold=filter_threshold)
    
        save_pkl(data=results_dict, path=implicit_output_file)
    results_dict = load_pkl(path=implicit_output_file)
    
    ambiguous_results = results_dict.get('ambiguous')
    unambiguous_results = results_dict.get('unambiguous')
    
    #########################
    #                       #
    # DATA ANNOTATION STAGE #
    #                       #
    #########################

    ## GENERATE DISAMBIGUATIONS ##
    if not os.path.isfile(selected_data_file):
        
        from pipeline.selection import data_selection
        
        selected_train_data, _ = data_selection(base_output_path=base_output_path,
                                                ABLATION_METHODS=ABLATION_METHODS, 
                                                ambiguous_results=ambiguous_results, 
                                                unambiguous_results=unambiguous_results, 
                                                correct_samples=correct_samples, 
                                                implicit_results=implicit_results,
                                                model=model, 
                                                args=args)
        
        if tokenizer is None:
            _, tokenizer = load_vllm_model_and_tokenizer(args, stage_index=STAGE_INDEX, load_model=False)
            
        data_module = make_supervised_data_module(tokenizer=tokenizer, dataset=selected_train_data, template_id=INFERENCE_TEMPLATE_ID)

        save_pkl(data=data_module, path=selected_data_file)

    data_module = load_pkl(path=selected_data_file)

    logger.info(f'Done stage {STAGE_INDEX}. Start training model.')


if __name__ == '__main__':
    main()