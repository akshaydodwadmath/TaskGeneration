from __future__ import division
# External imports
import json
import os
import torch

from torch.autograd import Variable
from tqdm import tqdm

from dataloader import load_input_file_orig, get_minibatch, shuffle_dataset
from karel.consistency import Simulator


def add_eval_args(parser):
    parser.add_argument('--eval_nb_ios', type=int,
                        default=5)
    parser.add_argument('--use_grammar', action="store_true")
    parser.add_argument("--val_nb_samples", type=int,
                        default=0,
                        help="How many samples to use to compute the accuracy."
                        "Default: %(default)s, for all the dataset")
    
def add_beam_size_arg(parser):
    parser.add_argument("--eval_batch_size", type=int,
                        default=8)
    parser.add_argument("--beam_size", type=int,
                        default=10,
                        help="Size of the beam search. Default %(default)s")
    parser.add_argument("--top_k", type=int,
                        default=5,
                        help="How many candidates to return. Default %(default)s")

def add_common_arg(parser):
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use the GPU to run the model")
    parser.add_argument("--log_frequency", type=int,
                        default=100,
                        help="How many minibatch to do before logging"
                        "Default: %(default)s.")
    
def evaluate_model(model_weights,
                   vocabulary_path,
                   dataset_path,
                   nb_ios,
                   nb_samples,
                   use_grammar,
                   output_path,
                   beam_size,
                   top_k,
                   batch_size,
                   use_cuda,
                   dump_programs):

    all_semantic_output_path = []
    
    res_dir = os.path.dirname(output_path)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for k in range(top_k):
        
        new_semantic_term = "semantic_top%d.txt" % (k+1)
        
        new_semantic_file_name = output_path + new_semantic_term
        

        all_semantic_output_path.append(new_semantic_file_name)
    program_dump_path = os.path.join(res_dir, "generated")
        
    # Load the vocabulary of the trained model
    dataset, vocab = load_input_file_orig(dataset_path, vocabulary_path)
    tgt_start = vocab["tkn2idx"]["<s>"]
    tgt_end = vocab["tkn2idx"]["m)"]
    tgt_pad = vocab["tkn2idx"]["<pad>"]

    simulator = Simulator(vocab["idx2tkn"])
    
    
    # Load the model
    if not use_cuda:
        # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/8
        # Is it failing?
        model = torch.load(model_weights, map_location=lambda storage, loc: storage)
    else:
        model = torch.load(model_weights)
        model.cuda()
        
    # And put it into evaluation mode
    model.eval()
    
    if use_grammar:
        syntax_checker = PySyntaxChecker(vocab["tkn2idx"], use_cuda)
        model.set_syntax_checker(syntax_checker)
        
        
    nb_semantic_correct = [0 for _ in range(top_k)]
    total_nb = 0
    
    
    dataset = shuffle_dataset(dataset, batch_size, randomize=False)
    
    
    for sp_idx in tqdm(range(0, len(dataset["sources"]), batch_size)):


        inp_grids, out_grids, \
        in_tgt_seq, in_tgt_seq_list, out_tgt_seq, \
        inp_worlds, out_worlds, \
        _, \
        inp_test_worlds, out_test_worlds = get_minibatch(dataset, sp_idx, batch_size,
                                                         tgt_start, tgt_end, tgt_pad,
                                                         nb_ios, shuffle=False, volatile_vars=True)
    
        #TODO: WHY?
        max_len = out_tgt_seq.size(1) + 10
        if use_cuda:
            inp_grids, out_grids = inp_grids.cuda(), out_grids.cuda()
            in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()
         
        #TODO: Understand code for dumping programs 
        if dump_programs:
            import numpy as np
            decoder_logit, syntax_logit = model(inp_grids, out_grids, in_tgt_seq, in_tgt_seq_list)
            if syntax_logit is not None and model.decoder.learned_syntax_checker is not None:
                syntax_logit = syntax_logit.cpu().data.numpy()
                for n in range(in_tgt_seq.size(0)):
                    decoded_dump_dir = os.path.join(program_dump_path, str(n + sp_idx))
                    if not os.path.exists(decoded_dump_dir):
                        os.makedirs(decoded_dump_dir)
                    seq = in_tgt_seq.cpu().data.numpy()[n].tolist()
                    seq_len = seq.index(0) if 0 in seq else len(seq)
                    file_name = str(n) + "_learned_syntax"
                    norm_logit = syntax_logit[n,:seq_len]
                    norm_logit = np.log(-norm_logit)
                    norm_logit = 1 / (1 + np.exp(-norm_logit))
                    np.save(os.path.join(decoded_dump_dir, file_name), norm_logit)
                    ini_state = syntax_checker.get_initial_checker_state()
                    file_name = str(n) + "_manual_syntax"
                    mask = syntax_checker.get_sequence_mask(ini_state, seq).squeeze().cpu().numpy()[:seq_len]
                    np.save(os.path.join(decoded_dump_dir, file_name), mask)
                    file_name = str(n) + "_diff"
                    diff = mask.astype(float) - norm_logit
                    diff = (diff + 1) / 2 # remap to [0,1]
                    np.save(os.path.join(decoded_dump_dir, file_name), diff)

        decoded = model.beam_sample(inp_grids, out_grids,
                                    tgt_start, tgt_end, max_len,
                                    beam_size, top_k)
        
        for batch_idx, (target, sp_decoded,
                        sp_input_worlds, sp_output_worlds,
                        sp_test_input_worlds, sp_test_output_worlds) in \
            enumerate(zip(out_tgt_seq.chunk(out_tgt_seq.size(0)), decoded,
                          inp_worlds, out_worlds,
                          inp_test_worlds, out_test_worlds)):

            total_nb += 1 #should be batch size * number of IOs
            target = target.cpu().data.squeeze().numpy().tolist()
            target = [tkn_idx for tkn_idx in target if tkn_idx != tgt_pad]
            
            
            if dump_programs:
                decoded_dump_dir = os.path.join(program_dump_path, str(batch_idx + sp_idx))
                if not os.path.exists(decoded_dump_dir):
                    os.makedirs(decoded_dump_dir)
                write_program(os.path.join(decoded_dump_dir, "target"), target, vocab["idx2tkn"])
                for rank, dec in enumerate(sp_decoded):
                    pred = dec[1]
                    ll = dec[0]
                    file_name = str(rank)+ " - " + str(ll)
                    write_program(os.path.join(decoded_dump_dir, file_name), pred, vocab["idx2tkn"])


            
            # Semantic matches
            for rank, dec in enumerate(sp_decoded):
                pred = dec[-1]
                print('pred', pred)
                parse_success, cand_prog = simulator.get_prog_ast(pred)
                if (not parse_success):
                    continue
                semantically_correct = True
                for (input_world, output_world) in zip(sp_input_worlds, sp_output_worlds):
                    res_emu = simulator.run_prog(cand_prog, input_world)
                    if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                        # This prediction is semantically incorrect.
                        semantically_correct = False
                        break
                if semantically_correct:
                    # Score for all the following ranks
                    for top_idx in range(rank, top_k):
                        nb_semantic_correct[top_idx] += 1
                    break
                
                
    for k in range(top_k):
        with open(str(all_semantic_output_path[k]), "w") as sem_res_file:
            sem_res_file.write(str(100*nb_semantic_correct[k]/total_nb))
            
    semantic_at_one = 100*nb_semantic_correct[0]/total_nb
    return semantic_at_one

def write_program(path, tkn_idxs, vocab):
    program_tkns = [vocab[tkn_idx] for tkn_idx in tkn_idxs]

    indent = 0
    is_new_line = False
    with open(path, "w") as target_file:
        for tkn in program_tkns:
            if tkn in ["m(", "w(", "i(", "e(", "r("]:
                indent += 4
                target_file.write("\n"+" "*indent)
                target_file.write(tkn + " ")
                is_new_line = False
            elif tkn in ["m)", "w)", "i)", "e)", "r)"]:
                if is_new_line:
                    target_file.write("\n"+" "*indent)
                indent -= 4
                target_file.write(tkn)
                if indent < 0:
                    indent = 0
                is_new_line = True
            elif tkn in ["REPEAT"]:
                if is_new_line:
                    target_file.write("\n"+" "*indent)
                    is_new_line = False
                target_file.write(tkn + " ")
            else:
                if is_new_line:
                    target_file.write("\n"+" "*indent)
                    is_new_line = False
                target_file.write(tkn + " ")
        target_file.write("\n")
