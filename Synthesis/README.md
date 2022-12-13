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

python3 main.py --train_file=../PreProcessing/train_betterquality.json --val_feature_file=../PreProcessing/featVectors_betterquality.json --batch_size=8  --nb_epochs=200 --n_domains 8 --shuffle_data --use_cuda --result_folder exps_batch8_better_quality --nb_samples 100 --topk 1

python3 eval_cmd.py --model_weights exps_K100_0.5Q/Weights/best.model --vocabulary data/new_vocab.vocab --feature_file_path ../PreProcessing/featVectors_14CT_15D_0.5Q.json --eval_batch_size 8 --output_path exps_K100_0.5Q/Results/ValSet_ --n_domains 100 --top_k 1 --use_cuda --train_file_path ../PreProcessing/train_14CT_15D_0.5Q.json --beam 64 --nb_samples 100



