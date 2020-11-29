#!/bin/bash
  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1

python run_language_modeling.py --output_dir=../../../../storage/logs/gpt2-scratch-1layer/ --model_type=gpt2 --tokenizer_name=microsoft/DialoGPT-small --config_name=../../../../storage/logs/only-config/config.json --do_train --train_data_file=../../../../storage/data/train.txt --do_eval --eval_data_file=../../../../storage/data/valid.txt --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=5e-5 --num_train_epochs=10 --line_by_line --save_steps=830 --save_total_limit=20 --evaluate_during_training --evaluation_strategy=steps --logging_steps=830 --eval_steps=830