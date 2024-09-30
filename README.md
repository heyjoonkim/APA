# Aligning Language Models to Explicitly Handle Ambiguity (EMNLP 2024)
Code for <b><i>Alignment with Perceived Ambiguity (APA)</i></b>

## Run
Run <code>sh scripts/main.sh</code>.
- <code>stage_0.sh</code> : select ambgiuous queries and build train data.
- <code>train.sh</code> : train model.
- <code>stage_1.sh</code> : evaluate trained model.

### Configurations
Change <code>configs/main.yaml</code>
- <code>model.name</code> : backbone
- <code>model.offload_path</code> : model offload path
- <code>model.cache_path</code> : huggingface cache path
- <code>path.data</code> : path to load dataset
- <code>path.output</code> : output path (logs, weights, ...)
- <code>dataset.name</code> : test dataset name
- <code>pipeline.stage_index</code> : set from 0 or 1
- <code>explicit.template_id</code> : explicit inference QA template
- <code>explicit.evaluation_method</code> : 'rouge' as default
- <code>explicit.correct_threshold</code> : generations with score above the threshold is evaluated as correct.
- <code>implicit.method_id</code> : how to measure INFOGAIN (default 0)
- <code>implicit.disambiguation_template_id</code> : template id for self-disambiguation
- <code>implicit.generation_template_id</code>
- <code>implicit.threshold</code> : threshold value to filter ambiguous queries
- <code>implicit.aggregate_method</code>
- <code>explanation.template_id</code> : template to generate explanations
- <code>generation.num_generations_per_prompt</code> : generation configs
- <code>generation.num_single_generation</code> : generation configs
- <code>generation.max_new_tokens</code> : generation configs
- <code>generation.temperature</code> : generation configs
- <code>ablation_methods</code> : data selection methods
- <code>train.num_train_epochs</code> : train configs (number of training epochs)
- <code>train.per_device_train_batch_size</code> : train configs (train batch size)
- <code>train.gradient_accumulation_steps</code> : train configs (gradient accumulation steps)
- <code>train.learning_rate</code> : train configs (learning rate)
