model='meta-llama/Llama-2-7b-hf'

accelerate launch --num_processes 4 --num_machines 1 \
    -m pipeline.train \
    --seed 1234 \
    --tf32 True \
    --bf16 True \
    --model_name_or_path $model \
    --offload_dir /YOUR/OFFLOAD/DIR/ \
    --cache_dir /YOUR/CACHE/DIR/ \
    --output_dir /YOUR/OUTPUT/DIR/ \
    --logging_level DEBUG \
    --dataset_name ambigqa \
    --explicit_template_id 0 \
    --implicit_method_id 0 \
    --explanation_template_id 0 \
    --ablation_methods 0 1 2 \
    --disambiguation_template_id 0 \
    --filter_threshold 0.1 \
    --stage_index 0 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate '1e-3' \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --optim "adamw_torch" \
    --lr_scheduler_type "cosine" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30000 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_bias "none" \
    --lora_task_type "CAUSAL_LM" \
    --lora_target_modules "q_proj" "v_proj" \
    --use_qlora True \
    --use_gptq False
                    