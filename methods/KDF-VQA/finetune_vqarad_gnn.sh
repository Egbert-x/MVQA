export CUDA_VISIBLE_DEVICES=1

ent_emb='ddb'
python3 -u finetune_VQA_RAD_gnn.py \
    --output_dir ./Results_finetune/VQA-RAD_biolinkbert_gnn_v1.1 \
    --pre_trained ./Results/only_pmc_clip_biolinkbert/vqa/checkpoint-11000  \
    --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --eval_steps 5 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --run_name SLAKE\
    --image_encoder "PMC_CLIP" \
    --bf16 True \
    --tf32 True \