model='meta-llama/Llama-2-7b-hf'

python -m pipeline.stage_0 \
    --config-path=../configs \
    --config-name=full_pipeline.yaml \
    seed=1234 \
    logging_level=DEBUG \
    model.name=$model