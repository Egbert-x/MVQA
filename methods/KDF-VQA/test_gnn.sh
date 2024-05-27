#export CUDA_VISIBLE_DEVICES=0
#python3 -u test_SLAKE_gnn.py \
#--output_dir ./Results_test/SLAKE_biolinkbgnn_gnn_v2 --ckp ./Results_finetune/SLAKE_biolinkbert_gnn_v2/vqa/checkpoint-53800 --image_encoder PMC_CLIP

export CUDA_VISIBLE_DEVICES=0
python3 -u test_gnn.py \
--output_dir ./Results_test/PMC-VQA_biolinkbert_gnn_v2.1 --ckp ./Results/pretrain_biolinkbert_gnn_v2.1/vqa/checkpoint-44200 --image_encoder PMC_CLIP

#export CUDA_VISIBLE_DEVICES=1
#python3 -u test_VQA_RAD_gnn.py \
#--output_dir ./Results_test/VQA-RAD_biolinkbert_gnn_v2 --ckp ./Results_finetune/VQA-RAD_biolinkbert_gnn_v2/vqa/checkpoint-25800 --image_encoder PMC_CLIP