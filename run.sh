pip install --upgrade huggingface_hub
huggingface-cli login
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

pip install deepspeed
# pip install flash-attn --no-build-isolation

# Inside LLaMA-Factory
python -m pip install --upgrade huggingface_hub && \
python -m pip install -e ".[torch,metrics]" && \
python -m pip install deepspeed && \
python -m pip install hf_transfer && \
huggingface-cli login

nohup torchrun src/train.py \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset my_data \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj\
    --output_dir ./eddie-v3 \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 1000 \
    --weight_decay 0.05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16 > output.log 2>&1 &


pm2 start "vllm serve ./merged_model --port 33271 --host 0.0.0.0" --name "sn35-vllm"
