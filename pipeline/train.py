import os
import logging
from time import time
from typing import Optional, List
from dataclasses import dataclass, field

import torch
from accelerate import Accelerator
from transformers import Trainer, HfArgumentParser, BitsAndBytesConfig, TrainingArguments, GPTQConfig
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForCausalLM


from utils.logging import logger_init
from utils.data import save_pkl, load_pkl
from models import load_model_and_tokenizer
from utils import seed_everything

logger = logging.getLogger(__name__)


####################################
#                                  #
# PARAMETER ARGUMENTS FOR TRAINING #
#                                  #
####################################

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    offload_dir: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    dataset_name: str = field(default='alpaca')

@dataclass
class CustomTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    remove_unused_columns: bool = field(default=False)
    logging_level: Optional[str] = field(default='INFO')
    explicit_template_id: int = field(default=0)
    implicit_method_id: int = field(default=0)
    disambiguation_template_id: int = field(default=0)
    filter_threshold: str = field(default=0)
    stage_index: int = field(default=0)
    logging_level: Optional[str] = field(default='INFO')
    ablation_methods: List[int] = field(default=None)
    explanation_template_id: int = field(default=0)
    
@dataclass
class LoraArguments:
    lora_task_type: str = field(default='CAUSAL_LM')
    inference_mode: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_target_modules: List[str] = field(default=None)
    use_qlora: bool = field(default=False)
    use_gptq: bool = field(default=True)

     

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    random_seed = training_args.seed   

    seed_everything(random_seed)

    accelerator = Accelerator()
    
    model_name = model_args.model_name_or_path
    model_name_for_dir = model_name.replace('/', '-')

    ## BASE PATH for outputs
    base_path = os.path.join(training_args.output_dir, model_name_for_dir, str(random_seed))

    ############################
    #                          #
    #       LOAD DATASET       #
    #                          #
    ############################

    EXPLICIT_TEMPLATE_ID = training_args.explicit_template_id
    IMPLICIT_METHOD_ID = training_args.implicit_method_id
    DISAMBIGUATION_TEMPLATE_ID = training_args.disambiguation_template_id
    STAGE_INDEX = training_args.stage_index
    THRESHOLD = training_args.filter_threshold
    EXPLANATION_TEMPLATE_ID = training_args.explanation_template_id
    
    ABLATION_METHODS = training_args.ablation_methods
    ablation_methods_str = '-'.join([str(method_id) for method_id in ABLATION_METHODS])
    
    selected_data_path = os.path.join(base_path, f'explicit_template_{EXPLICIT_TEMPLATE_ID}_implicit_method_{IMPLICIT_METHOD_ID}_template_{DISAMBIGUATION_TEMPLATE_ID}')
    assert os.path.isdir(selected_data_path), f'Selected data path does not exist : {selected_data_path}'

    selected_data_file = os.path.join(selected_data_path, f'3_selected_data_method_{ablation_methods_str}_threshold_{THRESHOLD}_stage_{STAGE_INDEX}.pkl')
    
    if 2 in ABLATION_METHODS:
        selected_data_file = os.path.join(selected_data_path, f'3_selected_data_method_{ablation_methods_str}_threshold_{THRESHOLD}_explanation_template_{EXPLANATION_TEMPLATE_ID}_stage_{STAGE_INDEX}.pkl')
    assert os.path.isfile(selected_data_file), f'Selected data file does not exist : {selected_data_file}'
    
    data_module = load_pkl(path=selected_data_file)
    logger.info(f'Loaded data module : {selected_data_file}')


    ###########################################
    #                                         #
    # TRAINING STAGE                          #
    # - train model with new annotated labels #
    #                                         #
    ###########################################

    ## SET OUTPUT DIRECTORY ##
    CONFIG_STR = f'method_{ablation_methods_str}_threshold-{THRESHOLD}_epoch-{int(training_args.num_train_epochs)}_batch-{training_args.per_device_train_batch_size}_accumulation-{training_args.gradient_accumulation_steps}_lr-{training_args.learning_rate}'
    if 2 in ABLATION_METHODS:
        CONFIG_STR = f'method_{ablation_methods_str}_threshold-{THRESHOLD}_exp-{EXPLANATION_TEMPLATE_ID}_epoch-{int(training_args.num_train_epochs)}_batch-{training_args.per_device_train_batch_size}_accumulation-{training_args.gradient_accumulation_steps}_lr-{training_args.learning_rate}'
    
    model_checkpoint_path = os.path.join(selected_data_path, CONFIG_STR, f'checkpoint_stage_{STAGE_INDEX}')

    with accelerator.main_process_first():
        logger.info(f'Checkpoint path does not exist. Train model. Generate output directory: {model_checkpoint_path}')
        os.makedirs(model_checkpoint_path, exist_ok=True)

    ## INIT LOGGER
    logger_level = training_args.logging_level
    logger_init(logger, model_checkpoint_path, logger_level=logger_level, save_as_file=True)

    ## save config ## 
    training_config_file = os.path.join(model_checkpoint_path, 'training_config.pkl')
    save_pkl(data=[model_args, data_args, training_args, lora_args], path=training_config_file)
    
    ## Train model to learn ambiguity ##
    ## Initialize PEFT Configs ##
    lora_config = LoraConfig(
                        r=lora_args.lora_r,
                        lora_alpha=lora_args.lora_alpha,
                        lora_dropout=lora_args.lora_dropout,
                        bias=lora_args.lora_bias,
                        inference_mode=False,
                        task_type=TaskType.CAUSAL_LM,
                        target_modules=lora_args.lora_target_modules,
                    )

    quantization_config = None
    if lora_args.use_qlora:
        logger.info('** Use QLoRa quantization.')
        quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16
                            )
    else:
        logger.info('** No quantization.')
    
    
    ## LOAD MODEL and TOKENIZER ##
    if STAGE_INDEX == 0:
        # load base model
        model, tokenizer = load_model_and_tokenizer(
                                                model_name=model_args.model_name_or_path,
                                                cache_path=training_args.cache_dir,
                                                offload_path=model_args.offload_dir,
                                                load_model=True,
                                                quantization_config=quantization_config)     
    else: 
        previous_stage_index = STAGE_INDEX - 1
        logger.info(f'Load trained model from stage {previous_stage_index}')
        previous_model_checkpoint_path = os.path.join(selected_data_path, CONFIG_STR, f'checkpoint_stage_{previous_stage_index}')
        logger.info(f'Load trained model from: {previous_model_checkpoint_path}')
        # load trained model from the previous stage
        model, tokenizer = load_model_and_tokenizer(
                                                model_name=previous_model_checkpoint_path,
                                                load_model=True,
                                                quantization_config=quantization_config)  

    if model is not None:
        model.config.use_cache=False
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    training_args.logging_dir = os.path.join(model_checkpoint_path, 'logs')

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    logger.info('Start training.')
    start_time = time()
    trainer.train()
    end_time = time()
    logger.info(f'Training time : {end_time - start_time} seconds.')
    
    if accelerator.is_main_process:
        logger.info('Save model and state only in main process.')
        trainer.save_model(output_dir=model_checkpoint_path)
        logger.info('Save 4-bit quantized model...')
        del trainer
        torch.cuda.empty_cache()

        logger.info('Load model in AutoPeftModelForCausalLM.')
        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_checkpoint_path,
            cache_dir=training_args.cache_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )  
        logger.info('Merge LoRA and base model and save.')
        # Merge LoRA and base model and save
        model = model.merge_and_unload()      
        logger.info('Save final model...')  
        start_time = time()
        model.save_pretrained(model_checkpoint_path)
        end_time = time()
        logger.info(f'Save time : {end_time - start_time} seconds.')
        
        logger.info('Remove adapter files.')
        file_list = os.listdir(model_checkpoint_path)
        for file in file_list:
            if 'adapter' in file:
                os.remove(os.path.join(model_checkpoint_path, file))

    logger.info(f'Done [Process {accelerator.process_index}].')


if __name__ == "__main__":
    train()