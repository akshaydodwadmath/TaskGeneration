# External imports
import json
import logging
import os
import random
import time


import argparse
from dataloader import load_input_file,get_minibatch, load_input_file_orig, shuffle_dataset
from train_helper import do_supervised_minibatch
from model import IOs2Seq
from evaluate import evaluate_model

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
signals = ["supervised", "rl", "beam_rl"]
use_grammar = False


class TrainSignal(object):
    SUPERVISED = "supervised"
    RL = "rl"
    BEAM_RL = "beam_rl"
    
def add_train_cli_args(parser):
    train_group = parser.add_argument_group("Training",
                                            description="Training options")
    train_group.add_argument('--signal', type=str,
                             choices=signals,
                             default=signals[0],
                             help="Where to get gradients from"
                             "Default: %(default)s")
    train_group.add_argument('--nb_ios', type=int,
                             default=5)
    train_group.add_argument('--nb_epochs', type=int,
                             default=2,
                             help="How many epochs to train the model for. "
                             "Default: %(default)s")
    train_group.add_argument('--optim_alg', type=str,
                             default='Adam',
                             choices=['Adam', 'RMSprop', 'SGD'],
                             help="What optimization algorithm to use. "
                             "Default: %(default)s")
    train_group.add_argument('--batch_size', type=int,
                             default=32,
                             help="Batch Size for the optimization. "
                             "Default: %(default)s")
    train_group.add_argument('--learning_rate', type=float,
                             default=1e-4,
                             help="Learning rate for the optimization. "
                             "Default: %(default)s")
    train_group.add_argument("--train_file", type=str,
                             default="data/1m_6ex_karel/train.json",
                             help="Path to the training data. "
                             " Default: %(default)s")
    train_group.add_argument("--val_file", type=str,
                             default="data/1m_6ex_karel/val.json",
                             help="Path to the validation data. "
                             " Default: %(default)s")
    train_group.add_argument("--vocab", type=str,
                             default="data/new_vocab.vocab",
                             help="Path to the output vocabulary."
                             " Default: %(default)s")
    train_group.add_argument("--nb_samples", type=int,
                             default=5,
                             help="Max number of samples to look at."
                             "If 0, look at the whole dataset.")
    train_group.add_argument("--result_folder", type=str,
                             default="exps/fake_run",
                             help="Where to store the results. "
                             " Default: %(default)s")
    train_group.add_argument("--init_weights", type=str,
                             default=None)
    train_group.add_argument("--use_grammar", action="store_true")
    train_group.add_argument('--beta', type=float,
                             default=1e-3,
                             help="Gain applied to syntax loss. "
                             "Default: %(default)s")
    train_group.add_argument("--val_frequency", type=int,
                             default=1,
                             help="Frequency (in epochs) of validation.")
                             
    train_group.add_argument("--use_cuda", action="store_true",
                        help="Use the GPU to run the model")
    train_group.add_argument("--log_frequency", type=int,
                        default=100,
                        help="How many minibatch to do before logging"
                        "Default: %(default)s.")
    train_group.add_argument("--save_to_txt", action="store_true",
                    help="Create data files with desried programs")

    rl_group = parser.add_argument_group("RL-specific training options")
    # rl_group.add_argument("--environment", type=str,
                          # choices=EnvironmentClasses.keys(),
                          # default="BlackBoxGeneralization",
                          # help="What type of environment to get a reward from"
                          # "Default: %(default)s.")
    # rl_group.add_argument("--reward_comb", type=str,
                          # choices=RewardCombinationFun.keys(),
                          # default="RenormExpected",
                          # help="How to aggregate the reward over several samples.")
    rl_group.add_argument('--nb_rollouts', type=int,
                          default=100,
                          help="When using RL,"
                          "how many trajectories to sample per example."
                          "Default: %(default)s")
    rl_group.add_argument('--rl_beam', type=int,
                          default=50,
                          help="Size of the beam when doing reward"
                          " maximization over the beam."
                          "Default: %(default)s")
    rl_group.add_argument('--rl_inner_batch', type=int,
                          default=2,
                          help="Size of the batch on expanded candidates")
    rl_group.add_argument('--rl_use_ref', action="store_true")
    
parser = argparse.ArgumentParser(
    description='Train a simple program synthesis model.')
add_train_cli_args(parser)
args = parser.parse_args()

#############################
# Admin / Bookkeeping stuff #
#############################
# Creating the results directory
result_dir = Path(args.result_folder)
if not result_dir.exists():
    os.makedirs(str(result_dir))
else:
    # The result directory exists. Let's check whether or not all of our
    # work has already been done.

    # The sign of all the works being done would be the model after the
    # last epoch, let's check if it's here
    last_epoch_model_path = result_dir / "Weights" / ("weights_%d.model" % (args.nb_epochs - 1))
    if last_epoch_model_path.exists():
        print("{} already exists -- skipping this training".format(last_epoch_model_path))
        #return
        
# Setting up the logs
log_file = result_dir / "logs.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=str(log_file),
    filemode='w'
)
train_loss_path = result_dir / "train_loss.json"
models_dir = result_dir / "Weights"
if not models_dir.exists():
    os.makedirs(str(models_dir))
    time.sleep(1)  # Let some time for the dir to be created
            
# Load-up the dataset TODO
if(args.save_to_txt):
    dataset, vocab = load_input_file(args.train_file, args.vocab)
else:
    dataset, vocab = load_input_file_orig(args.train_file, args.vocab)

#TODO
if use_grammar:
    syntax_checker = PySyntaxChecker(vocab["tkn2idx"], args.use_cuda)
    
vocabulary_size = len(vocab["tkn2idx"])

# Create the model
kernel_size = 3
conv_stack = [64, 64, 64]
fc_stack = [512]
tgt_embedding_size = 256
lstm_hidden_size = 256
nb_lstm_layers = 2
learn_syntax = False

#Need to setup paths
model = IOs2Seq(kernel_size, conv_stack, fc_stack,
                vocabulary_size, tgt_embedding_size,
                lstm_hidden_size, nb_lstm_layers,
                learn_syntax)
# Dump initial weights
path_to_ini_weight_dump = models_dir / "ini_weights.model"
with open(str(path_to_ini_weight_dump), "wb") as weight_file:
    torch.save(model, weight_file)                


if use_grammar:
    model.set_syntax_checker(syntax_checker)    
    
tgt_start = vocab["tkn2idx"]["<s>"]
tgt_end = vocab["tkn2idx"]["m)"]
tgt_pad = vocab["tkn2idx"]["<pad>"]

signal = args.signal

if signal == TrainSignal.SUPERVISED:
    # Create a mask to not penalize bad prediction on the padding
    weight_mask = torch.ones(vocabulary_size)
    weight_mask[tgt_pad] = 0
    # Setup the criterion
    loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
    
#TODO    
if args.use_cuda:
    model.cuda()
    if signal == TrainSignal.SUPERVISED:
        loss_criterion.cuda()
        
# Setup the optimizers
optimizer_cls = getattr(optim, args.optim_alg)
optimizer = optimizer_cls(model.parameters(),
                           lr=args.learning_rate)
                          
                          
#####################
# ################# #
# # Training Loop # #
# ################# #
#####################

losses = []
recent_losses = []
best_val_acc = np.NINF
batch_size = args.batch_size
for epoch_idx in range(0, args.nb_epochs):
    nb_ios_for_epoch = args.nb_ios
    # This is definitely not the most efficient way to do it but oh well
    dataset = shuffle_dataset(dataset, batch_size)
    for sp_idx in tqdm(range(0, len(dataset["sources"]), batch_size)):
    #for sp_idx in tqdm(range(0, 1, batch_size)):


        batch_idx = int(sp_idx/batch_size)
        optimizer.zero_grad()

        if signal == TrainSignal.SUPERVISED:
            inp_grids, out_grids, \
                in_tgt_seq, in_tgt_seq_list, out_tgt_seq, \
                _, _, _, _, _ = get_minibatch(dataset, sp_idx, batch_size,
                                              tgt_start, tgt_end, tgt_pad,
                                              nb_ios_for_epoch)
            #TODO
            if args.use_cuda:
                inp_grids, out_grids = inp_grids.cuda(), out_grids.cuda()
                in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()
            # if learn_syntax:
                # minibatch_loss = do_syntax_weighted_minibatch(model,
                                                              # inp_grids, out_grids,
                                                              # in_tgt_seq, in_tgt_seq_list,
                                                              # out_tgt_seq,
                                                              # loss_criterion, beta)
            # else:
            minibatch_loss = do_supervised_minibatch(model,
                                                     inp_grids, out_grids,
                                                     in_tgt_seq, in_tgt_seq_list,
                                                     out_tgt_seq, loss_criterion)
            recent_losses.append(minibatch_loss)
            
        optimizer.step()
        
        if (batch_idx % args.log_frequency == args.log_frequency-1 and len(recent_losses) > 0) or \
           (len(dataset["sources"]) - sp_idx ) < batch_size:
            logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (
                epoch_idx, batch_idx, sum(recent_losses)/len(recent_losses))
            )
            losses.extend(recent_losses)
            recent_losses = []
            # Dump the training losses
            with open(str(train_loss_path), "w") as train_loss_file:
                json.dump(losses, train_loss_file, indent=2)
                
    # Dump the weights at the end of the epoch
    path_to_weight_dump = models_dir / ("weights_%d.model" % epoch_idx)
    with open(str(path_to_weight_dump), "wb") as weight_file:
        # Needs to be in cpu mode to dump, otherwise will be annoying to load
        if args.use_cuda:
            model.cpu()
        torch.save(model, weight_file)
        if args.use_cuda:
            model.cuda()
    previous_weight_dump = models_dir / ("weights_%d.model" % (epoch_idx-1))
    if previous_weight_dump.exists():
        os.remove(str(previous_weight_dump))
    # Dump the training losses
    with open(str(train_loss_path), "w") as train_loss_file:
        json.dump(losses, train_loss_file, indent=2)

    logging.info("Done with epoch %d." % epoch_idx)
    
    
    if (epoch_idx+1) % args.val_frequency == 0 or (epoch_idx+1) == args.nb_epochs:
        # Evaluate the model on the validation set
        out_path = str(result_dir / ("eval/epoch_%d/val_.txt" % epoch_idx))
        val_acc = evaluate_model(str(path_to_weight_dump), args.vocab,
                                 args.val_file, 5, 0, use_grammar,
                                 out_path, 100, 50, batch_size,
                                 args.use_cuda, False)
        logging.info("Epoch : %d ValidationAccuracy : %f." % (epoch_idx, val_acc))
        if val_acc > best_val_acc:
            logging.info("Epoch : %d ValidationBest : %f." % (epoch_idx, val_acc))
            best_val_acc = val_acc
            path_to_weight_dump = models_dir / "best.model"
            with open(str(path_to_weight_dump), "wb") as weight_file:
                # Needs to be in cpu mode to dump, otherwise will be annoying to load
                if args.use_cuda:
                    model.cpu()
                torch.save(model, weight_file)
                if args.use_cuda:
                    model.cuda()
