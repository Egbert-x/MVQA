#export CUDA_VISIBLE_DEVICES=0
python3 -u test_triples.py \
    --output_dir ./Results_test/PMC-OA_triples \
    --ckp ../../../PMC-VQA/src/MedVInT_TD/VQA_lora_PMC_LLaMA_PMCCLIP/choice/checkpoint-4000
#    --ckp ./Results/PMC-OA/choice_training/checkpoint-7000

#python3 -u test_SLAKE_triples.py \
#    --output_dir ./Results_test/SLAKE_triples \
#    --ckp ./Results_finetune/SLAKE/checkpoint-10500
#    --ckp ../../../PMC-VQA/src/MedVInT_TD/VQA_lora_PMC_LLaMA_PMCCLIP/blank

#python3 -u test_VQA_RAD_triples.py \
#    --output_dir ./Results_test/VQA_RAD_triples \
#    --ckp ./Results_finetune/VQA_RAD/checkpoint-5500
#    --ckp ../../../PMC-VQA/src/MedVInT_TD/VQA_lora_PMC_LLaMA_PMCCLIP/blank