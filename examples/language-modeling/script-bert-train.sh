#!/bin/bash
  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:p100:1

python run_language_modeling.py --mlm --model_name_or_path=bert-base-uncased --train_data_file=../../../../../casino/pre-training/storage/data/for_pretraining/train.txt --eval_data_file=../../../../../casino/pre-training/storage/data/for_pretraining/valid.txt --do_train --do_eval --output_dir=../../../../../casino/pre-training/storage/logs/actual-15 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=5e-5 --num_train_epochs=20 --save_steps=2875 --save_total_limit=20 --evaluate_during_training --evaluation_strategy=steps --logging_steps=100 --eval_steps=100 --mlm_probability=0.15 --line_by_line
