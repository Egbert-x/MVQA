export CUDA_VISIBLE_DEVICES=0

ent_emb='ddb'
python3 -u train_gnn.py \
    --output_dir ./Results/pretrain_biolinkbert_gnn_v2.1 \
    --num_train_epochs 2 \
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
    --run_name VQA_LoRA_training \
    --image_encoder "PMC_CLIP" \
    --is_blank True \
    --tf32 True \
    --bf16 True \
#    --deepspeed ./ds_config/ds_config_zero2.json \ if deep_speed
#    --pretrained_model ./PMC_LLAMA_Model  \ if PMC-LLaMA, change this to your PMC-LLaMA model path
#    --pmcclip_pretrained "./models/pmc_clip/checkpoint.pt" \ if PMC-CLIP, change this to your PMC-CLIP model path