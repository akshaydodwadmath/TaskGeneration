Usage: python3 main.py --train_file=data/train.json --val_file=data/val.json --batch_size=128  --nb_epochs=100

Usage for generating selected data: python3 main.py --train_file=data/train_data.json --val_file=data/val_data.json --batch_size=16  --nb_epochs=5 --save_to_txt

Evaluation(Supervised):
python3 eval_cmd.py --model_weights exps/fake_run/Weights/best.model --vocabulary data/new_vocab.vocab --dataset data/test.json --eval_nb_ios 5 --eval_batch_size 8 
--output_path exps/Results/TestSet_ --beam_size 64 --top_k 10

python3 eval_cmd.py --model_weights exps/fake_run/Weights/best.model --vocabulary data/new_vocab.vocab --dataset data/val.json --eval_nb_ios 5 --eval_batch_size 8 
--output_path exps/Results/TestSet_ --beam_size 64 --top_k 10 --dump_programs 

python3 main.py --train_file=data/val_data.json --val_file=data/val_data.json --batch_size=1  --nb_epochs=100 --ndomains 20 --use_cuda

python3 eval_cmd.py --model_weights exps/fake_run/Weights/weights_199.model --vocabulary data/new_vocab.vocab --dataset ../PreProcessing/featVectors_wrkng.json --eval_batch_size 1 --output_path exps/Results_K8/SimpleSet_ --ndomains 8 --top_k 10 --use_cuda

exps_wrkng_K8:

python3 main.py --train_file=../PreProcessing/train_withactions_fixed16_2.json --val_file=data/val_data.json --batch_size=8  --nb_epochs=200 --ndomains 8 --use_cuda


python3 eval_cmd.py --model_weights exps_batch8_better_quality/Weights/best.model --vocabulary data/new_vocab.vocab --feature_file_path ../PreProcessing/featVectors_betterquality.json --eval_batch_size 8 --output_path exps_batch8_better_quality/Results/ValSet_ --n_domains 8 --top_k 13 --use_cuda --train_file_path ../PreProcessing/train_betterquality.json --beam 64 --nb_samples 100

python3 main.py --train_file=../PreProcessing/temp/N_10_2/train.json --val_bitmap_file=../PreProcessing/temp/N_10_2/bitmap.json --batch_size=8  --nb_epochs=200 --n_domains 10 --shuffle_data --result_folder exps_14CT_K10_0.5Q_evalwithQuality --top_k 10 --val_frequency 10 --use_grammar --use_cuda

python3 eval_cmd.py --model_weights exps_14CT_K10_0.5Q_withSyntax_withEval/Weights/best.model --vocabulary data/new_vocab.vocab --bitmap_file_path ../PreProcessing/temp/N_10_2/bitmap.json --eval_batch_size 1 --output_path exps_14CT_K10_0.5Q_withSyntax_withEval/Results_new_eval/ValSet_ --n_domains 10 --top_k 10 --train_file_path ../PreProcessing/temp/N_10_2/train.json --beam 64 --eval_quality --use_grammar --use_cuda


RL Training:
python3 main.py  --signal rl --learning_rate 1e-5 --init_weights exps_14CT_K10_0.5Q_evalwithQuality/Weights/weights_85.model --train_file ../PreProcessing/temp/N_10_2/bitmap.json --val_bitmap_file=../PreProcessing/temp/N_10_2/bitmap.json --result_folder exps_14CT_K10_0.5Q_evalwithQuality/reinforce_finetune_model85 --batch_size 8 --nb_rollouts 1 --nb_epochs 20 --n_domains 10 --top_k 10 --log_frequency 30 --num_tasks_iter 200 --val_frequency 5 --use_grammar --use_cuda


