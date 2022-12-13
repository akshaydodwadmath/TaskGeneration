# TaskGeneration
Preprocessing: 
python3 parser_code_to_codeType.py --input_code_file train_61CT.txt --code_type_file train_code_types.txt --json_data_file train_61CT_12D.json --ndomains 12 --json_featVectors_file featVectors_61CT_12D.json

Baseline:
python3 random_generator.py --data_dir data_temp/


python3 random_generator.py --data_dir generated --data_generator --num_codes_per_spec 1 --quality_threshold 0.5 --max_iterations 20000

python3 sampler.py --input_code_file ../Baseline/random_generator/baseline_14CT_810kcodes/baseline_14CT_810codes.txt --data_dir baseline_14CT_810kcodes

