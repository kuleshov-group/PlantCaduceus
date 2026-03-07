# Pre-training PlantCAD

[← Back to main README](../README.md)

For advanced users who want to pre-train PlantCAD or PlantCAD2 models from scratch or fine-tune on custom datasets.

**Requirements:**

- Large computational resources (multi-GPU recommended)
- WandB account for experiment tracking
- Custom genomic dataset in HuggingFace format

## Basic pre-training command

```bash
WANDB_PROJECT=PlantCAD python src/HF_pre_train.py \
    --do_train \
    --report_to wandb \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataset_name 'kuleshov-group/Angiosperm_16_genomes' \
    --soft_masked_loss_weights_train 0.1 \
    --soft_masked_loss_weights_evaluation 0.0 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 16 \
    --seed 32 \
    --save_strategy steps \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --logging_steps 10 \
    --max_steps 120000 \
    --warmup_steps 1000 \
    --save_total_limit 20 \
    --learning_rate 2E-4 \
    --lr_scheduler_type constant_with_warmup \
    --run_name test \
    --overwrite_output_dir \
    --output_dir "PlantCaduceus_train_1" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --tokenizer_name 'kuleshov-group/PlantCaduceus_l20' \
    --config_name 'kuleshov-group/PlantCaduceus_l20' \
    --log_level info
```

## Key parameters

- `dataset_name`: Your custom dataset or use our Angiosperm dataset
- `max_steps`: Total training steps (adjust based on dataset size)
- `learning_rate`: 2E-4 works well for most cases
- `Batch sizes`: Adjust based on your GPU memory
