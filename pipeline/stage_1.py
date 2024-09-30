#########################################
#   STAGE 1 PROCESS                     #
#   1. Explicit evaluation on test set  #
#########################################

import logging
import os
from time import time

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.logging import logger_init
from utils.data import save_pkl, load_pkl
from utils import seed_everything
from data.dataclass.utils import get_dataset_class
from models import load_vllm_model_and_tokenizer
from pipeline import get_vllm_param
from pipeline.explicit import explicit_ambiguity_detection
from pipeline.evaluation import analysis

logger = logging.getLogger(__name__)


@hydra.main(config_path='../configs/', config_name='main.yaml', version_base=None)
def main(args: DictConfig) -> None:

    random_seed = args.seed

    seed_everything(random_seed)

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
    STAGE_INDEX = 1
    INFERENCE_TEMPLATE_ID = args.pipeline.explicit.template_id
    IMPLICIT_METHOD_ID = args.pipeline.implicit.method_id
    DISAMBIGUATION_TEMPLATE_ID = args.pipeline.implicit.disambiguation_template_id
    EVALUATION_METHOD = args.pipeline.explicit.evaluation_method
    ABLATION_METHODS = args.ablation_methods
    ablation_methods_str = '-'.join([str(method_id) for method_id in ABLATION_METHODS])

    ### BASE PATH FOR OUTPUTS ###
    base_output_path = os.path.join(tmp_path, f'explicit_template_{INFERENCE_TEMPLATE_ID}_implicit_method_{IMPLICIT_METHOD_ID}_template_{DISAMBIGUATION_TEMPLATE_ID}')

   
    if not os.path.isdir(base_output_path):
        os.makedirs(base_output_path, exist_ok=True)
     
    ## save config ## 
    with open(os.path.join(base_output_path, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(args))
   

    ### set model checkpoint from the last stage ###
    filter_threshold = args.pipeline.implicit.threshold
    CONFIG_STR = f'method_{ablation_methods_str}_threshold-{filter_threshold}_epoch-{int(args.train.num_train_epochs)}_batch-{args.train.per_device_train_batch_size}_accumulation-{args.train.gradient_accumulation_steps}_lr-{args.train.learning_rate}'
    if 2 in ABLATION_METHODS:
        EXPLANATION_TEMPLATE_ID = args.pipeline.explanation.template_id
        CONFIG_STR = f'method_{ablation_methods_str}_threshold-{filter_threshold}_exp-{EXPLANATION_TEMPLATE_ID}_epoch-{int(args.train.num_train_epochs)}_batch-{args.train.per_device_train_batch_size}_accumulation-{args.train.gradient_accumulation_steps}_lr-{args.train.learning_rate}'
    
    
    previous_stage_index = STAGE_INDEX - 1
    previous_model_checkpoint_path = os.path.join(base_output_path, CONFIG_STR, f'checkpoint_stage_{previous_stage_index}')
    
    # stage 1   : EXPLICIT generation directory
    ambigqa_evaluation_results_file = os.path.join(previous_model_checkpoint_path, f'_results_stage_{STAGE_INDEX}_test_ambigqa_{EVALUATION_METHOD}.pkl')
    ambig_triviaqa_evaluation_results_file = os.path.join(previous_model_checkpoint_path, f'_results_stage_{STAGE_INDEX}_test_ambig_triviaqa_{EVALUATION_METHOD}.pkl')
    situated_qa_geo_evaluation_results_file = os.path.join(previous_model_checkpoint_path, f'_results_stage_{STAGE_INDEX}_test_situated_qa_geo_{EVALUATION_METHOD}.pkl')
    situated_qa_temp_evaluation_results_file = os.path.join(previous_model_checkpoint_path, f'_results_stage_{STAGE_INDEX}_test_situated_qa_temp_{EVALUATION_METHOD}.pkl')
    ambig_web_questions_evaluation_results_file = os.path.join(previous_model_checkpoint_path, f'_results_stage_{STAGE_INDEX}_test_ambig_web_questions_{EVALUATION_METHOD}.pkl')
    ambig_freebase_qa_evaluation_results_file = os.path.join(previous_model_checkpoint_path, f'_results_stage_{STAGE_INDEX}_test_ambig_freebase_qa_{EVALUATION_METHOD}.pkl')
   
    
    ## INIT LOGGER
    logger_level = args.logging_level
    logger_init(logger, previous_model_checkpoint_path, logger_level=logger_level, save_as_file=True)

    ############################
    #                          #
    #       LOAD DATASET       #
    #                          #
    ############################
    
    ### AmbigQA ###
    ambigqa_dataset_path = os.path.join(args.path.data, 'ambigqa')    
    dataset_class = get_dataset_class(name='ambigqa', data_path=ambigqa_dataset_path, seed=random_seed)
    dataset_dict = dataset_class.get_dataset()
    ambigqa_validation_dataset = dataset_dict.get('validation')
    ambigqa_test_dataset = dataset_dict.get('test')

    if ambigqa_test_dataset is None:
        logger.info(f'No test set for ambigqa. Use validation set instead.')
        ambigqa_test_dataset = ambigqa_validation_dataset

    logger.info(f'Load AmbigQA: {len(ambigqa_test_dataset)} samples.')

    ### AmbigTriviaQA ###
    ambig_triviaqa_dataset_path = os.path.join(args.path.data, 'ambig_trivia_qa')
    dataset_class = get_dataset_class(name='ambig_trivia_qa', data_path=ambig_triviaqa_dataset_path, seed=random_seed)
    dataset_dict = dataset_class.get_dataset()
    ambig_triviaqa_test_dataset = dataset_dict.get('test')
    logger.info(f'Load AmbigTriviaQA: {len(ambig_triviaqa_test_dataset)} samples.')
    
    ### SituatedQA (Geo) ###
    situated_qa_geo_dataset_path = os.path.join(args.path.data, 'situated_qa_geo')
    dataset_class = get_dataset_class(name='situated_qa_geo', data_path=situated_qa_geo_dataset_path, seed=random_seed)
    dataset_dict = dataset_class.get_dataset()
    situated_qa_geo_test_dataset = dataset_dict.get('test')
    logger.info(f'Load SituatedQA(Geo): {len(situated_qa_geo_test_dataset)} samples.')
    
    ### SituatedQA (Temp) ###
    situated_qa_temp_dataset_path = os.path.join(args.path.data, 'situated_qa_temp')
    dataset_class = get_dataset_class(name='situated_qa_temp', data_path=situated_qa_temp_dataset_path, seed=random_seed)
    dataset_dict = dataset_class.get_dataset()
    situated_qa_temp_test_dataset = dataset_dict.get('test')
    logger.info(f'Load SituatedQA(Temp): {len(situated_qa_temp_test_dataset)} samples.')
    
    ### AmbigWebQuestions ###
    ambig_web_questions_dataset_path = os.path.join(args.path.data, 'ambig_web_questions')
    dataset_class = get_dataset_class(name='ambig_web_questions', data_path=ambig_web_questions_dataset_path, seed=random_seed)
    dataset_dict = dataset_class.get_dataset()
    ambig_web_questions_test_dataset = dataset_dict.get('test')
    logger.info(f'Load AmbigWebQuestions: {len(ambig_web_questions_test_dataset)} samples.')
    
    ### AmbigFreebaseQA ###
    ambig_freebase_qa_dataset_path = os.path.join(args.path.data, 'ambig_freebase_qa')
    dataset_class = get_dataset_class(name='ambig_freebase_qa', data_path=ambig_freebase_qa_dataset_path, seed=random_seed)
    dataset_dict = dataset_class.get_dataset()
    ambig_freebase_qa_test_dataset = dataset_dict.get('test')
    logger.info(f'Load AmbigFreebaseQA: {len(ambig_freebase_qa_test_dataset)} samples.')

    ### INITIALIZE MODEL ###
    model = None
    
    ##################################################
    #                                                #
    # GET VLLM GENERATION CONFIGS (Greedy, Sampling) #
    #                                                #
    ##################################################

    params_dict = get_vllm_param(args)
    greedy_params = params_dict.get('greedy')

    ####################################
    #                                  #
    # EXPLICIT EVALUATION ON TEST SETS #
    #                                  #
    ####################################
    
                
    if not os.path.isfile(ambigqa_evaluation_results_file):
        if model is None:
            model, _ = load_vllm_model_and_tokenizer(args, stage_index=STAGE_INDEX, previous_model_checkpoint_path=previous_model_checkpoint_path)
        
        logger.info(f'Initial evaluation on AmbigQA test set @ stage {STAGE_INDEX}.')
        start_time = time()
        evaluation_results = explicit_ambiguity_detection(
                                                        model=model,
                                                        dataset=ambigqa_test_dataset,
                                                        template_id=INFERENCE_TEMPLATE_ID,
                                                        evaluation_method=args.pipeline.explicit.evaluation_method,
                                                        stage_index=STAGE_INDEX,
                                                        correct_threshold=args.pipeline.explicit.correct_threshold,
                                                        greedy_params=greedy_params,
                                                        analysis=True,
                                                        dataset_name='ambigqa',
                                                    )
        end_time = time()
        save_pkl(data=evaluation_results, path=ambigqa_evaluation_results_file)
        logger.info(f'Done. Total time : {end_time - start_time} sec.')
    else:
        logger.info(f'Initial evalution results already exists @ stage {STAGE_INDEX}.')

    # score, is_correct, prediction
    evaluation_results = load_pkl(path=ambigqa_evaluation_results_file)
    
    logger.info(f'Results on AmbigQA test set in stage {STAGE_INDEX}.')
    analysis(dataset=evaluation_results, logger=logger)

    #############################################
    #                                           #
    #               AmbigTriviaQA               #
    #                                           #
    #############################################
            
    if STAGE_INDEX == 1 and not os.path.isfile(ambig_triviaqa_evaluation_results_file):
        
        if model is None:
            model, _ = load_vllm_model_and_tokenizer(args, stage_index=STAGE_INDEX, previous_model_checkpoint_path=previous_model_checkpoint_path)
            
        logger.info(f'Initial evaluation on AmbigTriviaQA test set @ stage {STAGE_INDEX}.')
        start_time = time()
        evaluation_results = explicit_ambiguity_detection(
                                                        model=model,
                                                        dataset=ambig_triviaqa_test_dataset,
                                                        template_id=INFERENCE_TEMPLATE_ID,
                                                        evaluation_method=args.pipeline.explicit.evaluation_method,
                                                        stage_index=STAGE_INDEX,
                                                        correct_threshold=args.pipeline.explicit.correct_threshold,
                                                        greedy_params=greedy_params,
                                                        analysis=True,
                                                        dataset_name='ambig_trivia_qa'
                                                    )
        end_time = time()
        save_pkl(data=evaluation_results, path=ambig_triviaqa_evaluation_results_file)
        logger.info(f'Done. Total time : {end_time - start_time} sec.')
    else:
        logger.info(f'Initial evalution results already exists @ stage {STAGE_INDEX}.')

    # score, is_correct, prediction
    evaluation_results = load_pkl(path=ambig_triviaqa_evaluation_results_file)
    
    logger.info(f'Results on AmbigTriviaQA test set in stage {STAGE_INDEX}.')
    analysis(dataset=evaluation_results, logger=logger)
    
    ################################################
    #                                              #
    #               SituatedQA (Geo)               #
    #                                              #
    ################################################
                
    if STAGE_INDEX == 1 and not os.path.isfile(situated_qa_geo_evaluation_results_file):
        
        if model is None:
            model, _ = load_vllm_model_and_tokenizer(args, stage_index=STAGE_INDEX, previous_model_checkpoint_path=previous_model_checkpoint_path)
            
        logger.info(f'Initial evaluation on SituatedQA Geo test set @ stage {STAGE_INDEX}.')
        start_time = time()
        evaluation_results = explicit_ambiguity_detection(
                                                        model=model,
                                                        dataset=situated_qa_geo_test_dataset,
                                                        template_id=INFERENCE_TEMPLATE_ID,
                                                        evaluation_method=args.pipeline.explicit.evaluation_method,
                                                        stage_index=STAGE_INDEX,
                                                        correct_threshold=args.pipeline.explicit.correct_threshold,
                                                        greedy_params=greedy_params,
                                                        analysis=True,
                                                        dataset_name='situated_qa_geo'
                                                    )
        end_time = time()
        save_pkl(data=evaluation_results, path=situated_qa_geo_evaluation_results_file)
        logger.info(f'Done. Total time : {end_time - start_time} sec.')
    else:
        logger.info(f'Initial evalution results already exists @ stage {STAGE_INDEX}.')

    # score, is_correct, prediction
    evaluation_results = load_pkl(path=situated_qa_geo_evaluation_results_file)
    
    logger.info(f'Results on SituatedQA Geo test set in stage {STAGE_INDEX}.')
    analysis(dataset=evaluation_results, logger=logger)\
    
    #################################################
    #                                               #
    #               SituatedQA (Temp)               #
    #                                               #
    #################################################
            
    if STAGE_INDEX == 1 and not os.path.isfile(situated_qa_temp_evaluation_results_file):
        
        if model is None:
            model, _ = load_vllm_model_and_tokenizer(args, stage_index=STAGE_INDEX, previous_model_checkpoint_path=previous_model_checkpoint_path)
            
        logger.info(f'Initial evaluation on SituatedQA Temp test set @ stage {STAGE_INDEX}.')
        start_time = time()
        evaluation_results = explicit_ambiguity_detection(
                                                        model=model,
                                                        dataset=situated_qa_temp_test_dataset,
                                                        template_id=INFERENCE_TEMPLATE_ID,
                                                        evaluation_method=args.pipeline.explicit.evaluation_method,
                                                        stage_index=STAGE_INDEX,
                                                        correct_threshold=args.pipeline.explicit.correct_threshold,
                                                        greedy_params=greedy_params,
                                                        analysis=True,
                                                        dataset_name='situated_qa_temp'
                                                    )
        end_time = time()
        save_pkl(data=evaluation_results, path=situated_qa_temp_evaluation_results_file)
        logger.info(f'Done. Total time : {end_time - start_time} sec.')
    else:
        logger.info(f'Initial evalution results already exists @ stage {STAGE_INDEX}.')

    # score, is_correct, prediction
    evaluation_results = load_pkl(path=situated_qa_temp_evaluation_results_file)
    
    logger.info(f'Results on SituatedQA Temp test set in stage {STAGE_INDEX}.')
    analysis(dataset=evaluation_results, logger=logger)

    #################################################
    #                                               #
    #               AmbigWebQuestions               #
    #                                               #
    #################################################
                
    if STAGE_INDEX == 1 and not os.path.isfile(ambig_web_questions_evaluation_results_file):
        
        if model is None:
            model, _ = load_vllm_model_and_tokenizer(args, stage_index=STAGE_INDEX, previous_model_checkpoint_path=previous_model_checkpoint_path)
            
        logger.info(f'Initial evaluation on AmbigWebQuestions test set @ stage {STAGE_INDEX}.')
        start_time = time()
        evaluation_results = explicit_ambiguity_detection(
                                                        model=model,
                                                        dataset=ambig_web_questions_test_dataset,
                                                        template_id=INFERENCE_TEMPLATE_ID,
                                                        evaluation_method=args.pipeline.explicit.evaluation_method,
                                                        stage_index=STAGE_INDEX,
                                                        correct_threshold=args.pipeline.explicit.correct_threshold,
                                                        greedy_params=greedy_params,
                                                        analysis=True,
                                                        dataset_name='ambig_web_questions'
                                                    )
        end_time = time()
        save_pkl(data=evaluation_results, path=ambig_web_questions_evaluation_results_file)
        logger.info(f'Done. Total time : {end_time - start_time} sec.')
    else:
        logger.info(f'Initial evalution results already exists @ stage {STAGE_INDEX}.')

    # score, is_correct, prediction
    evaluation_results = load_pkl(path=ambig_web_questions_evaluation_results_file)
    
    logger.info(f'Results on AmbigWebQuestions test set in stage {STAGE_INDEX}.')
    analysis(dataset=evaluation_results, logger=logger)

    ###############################################
    #                                             #
    #               AmbigFreebaseQA               #
    #                                             #
    ###############################################
                
    if STAGE_INDEX == 1 and not os.path.isfile(ambig_freebase_qa_evaluation_results_file):
        
        if model is None:
            model, _ = load_vllm_model_and_tokenizer(args, stage_index=STAGE_INDEX, previous_model_checkpoint_path=previous_model_checkpoint_path)
            
        logger.info(f'Initial evaluation on AmbigFreebaseQA test set @ stage {STAGE_INDEX}.')
        start_time = time()
        evaluation_results = explicit_ambiguity_detection(
                                                        model=model,
                                                        dataset=ambig_freebase_qa_test_dataset,
                                                        template_id=INFERENCE_TEMPLATE_ID,
                                                        evaluation_method=args.pipeline.explicit.evaluation_method,
                                                        stage_index=STAGE_INDEX,
                                                        correct_threshold=args.pipeline.explicit.correct_threshold,
                                                        greedy_params=greedy_params,
                                                        analysis=True,
                                                        dataset_name='ambig_freebase_qa'
                                                    )
        end_time = time()
        save_pkl(data=evaluation_results, path=ambig_freebase_qa_evaluation_results_file)
        logger.info(f'Done. Total time : {end_time - start_time} sec.')
    else:
        logger.info(f'Initial evalution results already exists @ stage {STAGE_INDEX}.')

    # score, is_correct, prediction
    evaluation_results = load_pkl(path=ambig_freebase_qa_evaluation_results_file)
    
    logger.info(f'Results on AmbigFreebaseQA test set in stage {STAGE_INDEX}.')
    analysis(dataset=evaluation_results, logger=logger)
  
    
if __name__ == '__main__':
    main()
       