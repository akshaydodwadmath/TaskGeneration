Usage: python3 main.py --train_file=data/train.json --val_file=data/val.json --batch_size=128  --nb_epochs=100

Usage for generating selected data: python3 main.py --train_file=data/train_data.json --val_file=data/val_data.json --batch_size=16  --nb_epochs=5 --save_to_txt

Evaluation(Supervised):
python3 eval_cmd.py --model_weights exps/fake_run/Weights/best.model --vocabulary data/new_vocab.vocab --dataset data/test.json --eval_nb_ios 5 --eval_batch_size 8 
--output_path exps/Results/TestSet_ --beam_size 64 --top_k 10

python3 eval_cmd.py --model_weights exps/fake_run/Weights/best.model --vocabulary data/new_vocab.vocab --dataset data/val.json --eval_nb_ios 5 --eval_batch_size 8 
--output_path exps/Results/TestSet_ --beam_size 64 --top_k 10 --dump_programs 
