python3 -u finetune_triples.py \
    --Train_csv_path '../../../PMC-VQA/Slake1.0_new/train.csv' \
    --Eval_csv_path '../../../PMC-VQA/Slake1.0_new/validate.csv' \
    --output_dir ./Results_finetune_triples/SLAKE_triples \
    --run_name SLAKE_triples \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
#    --deepspeed ./ds_config/ds_config_zero2.json \
#    --checkpointing false \
