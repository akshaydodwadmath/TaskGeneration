# TaskGeneration
Preprocessing: 
python3 parser_code_to_codeType.py --input_code_file train_61CT.txt --code_type_file train_code_types.txt --json_data_file train_61CT_12D.json --ndomains 12 --json_featVectors_file featVectors_61CT_12D.json

Baseline:
python3 random_generator.py --num_codes_per_spec 10 --data_dir baseline_K10_1120F/ --min_index 0 --max_index 14


Baseline as dataset generator:
python3 random_generator.py --data_dir generated --data_generator --num_codes_per_spec 1 --quality_threshold 0.5 --max_iterations 20000 --min_index 0 --max_index 14

Sampler:
python3 sampler.py --input_code_file ../Baseline/random_generator/baseline_14CT_810kcodes/baseline_14CT_810codes.txt --data_dir baseline_14CT_810kcodes

Neural Task Synthesizer:
python3 quality_check.py --input_code_file Full_im_14CT_q0.5_K10_all_updated.txt --result_folder results_im_14CT_q0.5_K10_all_updated --n_domains 10 --top_k 5 




