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


python3 eval_cmd.py --model_weights exps_wrkng_K8/fake_run/Weights/weights_110.model --vocabulary data/new_vocab.vocab --dataset ../PreProcessing/featVectors.json --eval_batch_size 8 --output_path exps_wrkng_K8/Results_K8/SimpleSet_ --ndomains 8 --top_k 10 --use_cuda --train_file_path ../PreProcessing/train_withactions_fixed16_2.json 
