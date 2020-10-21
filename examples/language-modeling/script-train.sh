#!/bin/bash
  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1

python run_language_modeling.py --output_dir=../../../../storage/logs/dgpt-ft-s-analysis/ --model_type=gpt2 --model_name_or_path=microsoft/DialoGPT-small --do_train --train_data_file=../../../../storage/data/train.txt --do_eval --eval_data_file=../../../../storage/data/valid.txt --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=5e-5 --num_train_epochs=100 --line_by_line --save_steps=20000 --save_total_limit=20 --evaluate_during_training --evaluation_strategy=steps --logging_steps=100 --eval_steps=100