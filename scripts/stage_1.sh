model='meta-llama/Llama-2-7b-hf'

python -m pipeline.stage_1 \
    --config-path=../configs \
    --config-name=full_pipeline.yaml \
    seed=1234 \
    logging_level=DEBUG \
    model.name=$model \
    train.num_train_epochs=3 \
    train.learning_rate='1e-3'
