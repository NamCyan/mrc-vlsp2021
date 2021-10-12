#sketchy module

python3 ./run_cls.py \
    --model_type phobert_large \
    --do_eval \
    --do_train \
    --do_lower_case \
    --train_file ./data/VLSP_data/VLSP_train_split_wseg.json \
    --dev_file ./data/VLSP_data/VLSP_dev_split_wseg.json \
    --overwrite_output_dir \
    --max_seq_len 256 \
    --per_gpu_train_batch_size=4   \
    --per_gpu_eval_batch_size=4  \
    --gradient_accumulation_steps 8 \
    --warmup_steps=128 \
    --weight_decay 1e-3 \
    --lr 2e-5 \
    --epochs 10 \
    --output_dir result/cls_squad2_phobert_lr2e-5_len256_bs32_ep2_wm814 \
    --eval_all_checkpoints \
    --save_steps 1000 \
    --logging_steps 500
