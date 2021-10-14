python3 ./run_single_model.py \
<<<<<<< HEAD
    --model_type xlm_roberta_large \
=======
    --model_type xlm_roberta_mixlayer_large \
>>>>>>> 3594862dc203105b34527ec6cd9b02754159c7fb
    --do_eval \
    --version_2_with_negative \
    --train_file ../data/VLSP_data/VLSP_train_split.json \
    --predict_file ../data/VLSP_data/VLSP_dev_split.json \
    --learning_rate 2e-5 \
    --weight_decay 1e-3 \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --doc_stride 64 \
    --max_query_length=32 \
    --per_gpu_train_batch_size=2 \
    --per_gpu_eval_batch_size=2 \
    --gradient_accumulation_steps 8 \
    --warmup_steps=128 \
    --output_dir result/av_xlm_roberta_lr2e-5_len256_bs16_ep2_wm814 \
    --do_lower_case \
    --verbose_logging \
    --eval_all_checkpoints \
    --save_steps 2000 \
    --logging_steps 2500 \
    --n_best_size 20 \
    --max_answer_length=40
