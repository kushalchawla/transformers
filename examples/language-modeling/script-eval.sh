#!/bin/bash
  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:p100:1

python run_language_modeling.py --output_dir=../../../../storage/logs/run-eval/  --model_type=gpt2 --model_name_or_path=../../../../storage/logs/gpt2-scratch-1layer/checkpoint-6640 --tokenizer_name=../../../../storage/logs/gpt2-scratch-1layer --do_eval --eval_data_file=../../../../storage/data/test.txt --per_device_eval_batch_size=1 --line_by_line