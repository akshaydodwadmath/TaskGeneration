# Towards Synthesizing Qualitative and Diverse Programs for Block-Based Visual Programming 

This repository contains the code used for the project Towards Synthesizing Qualitative and Diverse
Programs for Block-Based Visual Programming.

## Requirements
I recommend installing this code into a virtual environment. In order to run
the code, you first need to install pytorch, following the instructions from
[the pytorch website](http://pytorch.org/). Once this is done, you can install
this package and its dependencies by running:

```bash
pip install cython
python setup.py install
```

## Code Structure
The code to run neural model can be found in Synthesis folder, while for the baseline model it is inside Baseline/random_generator. Also, the code for generating visual puzzles from programs is inside NeuralTaskSynthesizer folder.

## Neural Model- Training
* `--kernel_size`, `--conv_stack`, `--fc_stack`, `--tgt_embedding_size`
  are flags to specify the architecture of the model to learn. See
  `Synthesis/model.py` to see how they are used.
* `--use_grammar` makes the model use the handwritten syntax checker, found in
  `Synthesis/syntax/checker.pyx`. `--learn_syntax` adds a Syntax neural model that
  attempts to learn a syntax checker, jointly with the rest of the model. The
  importance of this objective is controlled by the `--beta` parameter.
* `--signal` allows to choose the loss, between `supervised`, and `rl`. Supervised
  attempts to reproduce the ground truth program, while `rl` try to maximize
  expected rewards. In order to be able to fit
  experiments in a single GPU, you may need to adjust `--nb_rollouts` (how many
  samples are taken from the model to estimate a gradient when using `rl`). There
  is also the `--rl_inner_batch` option that splits the computation of a batch into
  several minibatches that are separately evaluated before doing a gradient
  step.
* `--optim_alg` chooses the optimization algorithm used, `--batch_size` allows
  to choose the size of the mini batches. `--learning_rate` adjusts the learning
  rate. `--init_weights` can be used to specify a '.model' file from which to
  load weights.
* `--train_file` specify the json file where to look for the training samples
  and `--val_file` indicates a validation set. The validation set is used to
  keep track of the best model seen so far, so as to perform early stopping. The
  `--vocab` file is there to give a correspondence between tokens and indices in
  the learned predictions. Setting `--nb_samples` allows to train on only part
  of the dataset (0, the default, trains on the whole dataset.).
  `--result_folder` allows to indicate where the results of the experiment
  should be stored. Changing `--val_frequency` allows to evaluate accuracy on
  the validation set less frequently. 
* Specify `--use_cuda` to run everything on a GPU. You can use the
  `CUDA_VISIBLE_DEVICES` to run on a specific GPU.

```bash
# To run the neural model
# Train a simple supervised model, using the handcoded syntax checker
main.py   --kernel_size 3 \
          --conv_stack "64,64,64" \
          --fc_stack "512" \
          --tgt_embedding_size 256 \
          \
          --signal supervised \
          --nb_epochs 100 \
          --optim_alg Adam \
          --batch_size 128 \
          --learning_rate 1e-4 \
          \
          --train_file data/train.json \
          --val_file data/val.json \
          --vocab data/new_vocab.vocab \
          --result_folder exps/supervised_use_grammar \
          \
          --use_grammar \
          \
          --use_cuda

# Use a pretrained model, to fine-tune it using simple Reinforce
# Change the --environment flag if you want to use a reward including performance.
main.py  --signal rl \
         --environment BlackBoxGeneralization \
         --nb_rollouts 100 \
         \
         --init_weights exps/supervised_use_grammar/Weights/best.model \
         --nb_epochs 5 \
         --optim_alg Adam \
         --learning_rate 1e-5 \
         --batch_size 16 \
         \
         --train_file data/train.json \
         --val_file data/val.json \
         --vocab data/new_vocab.vocab \
         --result_folder exps/reinforce_finetune \
         \
         --use_grammar \
         \
         --use_cuda
```
## Neural Model- Evaluation
The evaluation command is fairly similar. Any flags non-specified has the same
role as for the `main.py` command. The relevant file is `Synthesis/evaluate.py`.

* `--model_weights` should point to the model to evaluate.
* `--dataset` should point to the json file containing the dataset you want to
  evaluate against.
* `--output_path` points to where the results should be written. This should be
  a prefix for all the names of the files that will be generated, followed 
* `--dump_programs` can be used to investigate by dumping the programs returned
  by the model.
* `--val_nb_samples` is analogous to `--nb_samples`, can be used to do
  evaluation on only part of the dataset.
* `--eval_batch_size` specifies the batch size to use during decoding. This
  doesn't affect accuracies and batching operations only allows to go faster.
* `--beam_size` controls the size of the beam search to run when decoding the
  programs and `--top_k` should be the largest integer for which the accuracies
  should be computed.

This will generate a set of files. If `--dump_programs` is passed, the `--top_k`
most likely programs for each element of the dataset will be dumped, with their
rank and their log-probability in the `generated` subfolder. This will also
include the reference program, under the name `target`.

```bash
# Evaluate a trained model on the test set
eval_cmd.py --model_weights exps/rl_finetune/Weights/best.model \
            \
            --vocabulary data/new_vocab.vocab \
            --dataset data/test.json \
            --eval_batch_size 8 \
            --output_path exps/rl_finetune/Results/TestSet_ \
            \
            --beam_size 64 \
            --top_k 10 \
            --use_grammar \
            \
            --use_cuda
```

## Baseline Model- Generation
* `--num_codes_per_spec` specifies the number of codes to be generated
  per defined specification.`--min_index` and `--max_index`, define
  the specifications for which codes need to be generated.

```bash
# Generate codes using baseline:
random_generator.py --num_codes_per_spec 10 \
                    --data_dir baseline/Results/Generated_ \
                    --min_index 0 \
                    --max_index 14
```
## Code to Task Generation
* `--input_code_file` specifies the file with the set of codes.
  `--n_domains` define the number of codes available for each
  specification.

```bash
# Generate tasks using codes:
quality_check.py --input_code_file exps/generated_codes/codes.txt \
                    --n_domains 10 
```




