from __future__ import division
# External imports
import json
import os
import torch

from torch.autograd import Variable
from tqdm import tqdm

from dataloader import load_input_file, get_minibatch, shuffle_dataset
from karel.consistency import Simulator
from preprocessing.parser_code_to_codeType import getBitmapVector
import pyximport
pyximport.install()
from syntax.checker import PySyntaxChecker

from src.neural_code2task_syn.task_synthesizer import obtain_karel_saturation_score_for_code
from src.karel_data_converters.converter_format_iclr18_to_karelgym import \
    iclr18_codejson_to_karelgym_codejson
from src.karel_emulator.code import Code

import numpy as np
def add_eval_args(parser):
    parser.add_argument('--use_grammar', action="store_true")
    parser.add_argument('--extra_info', action="store_true")
    parser.add_argument('--eval_quality', action="store_true")
    parser.add_argument('--n_domains', type=int,
                        default=20,
                        help="Number of domains for target encoder. "
                        "Default: %(default)s")
    
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
                   bitmap_file_path,
                   train_file_path,
                   n_domains,
                   use_grammar,
                   output_path,
                   beam_size,
                   top_k,
                   batch_size,
                   use_cuda,
                   dump_programs,
                   eval_quality,
                   extra_info):
    
    res_dir = os.path.dirname(output_path)
    
    text = ""
    text_path = os.path.join(res_dir, "{}.txt".format('model_generated'))
    
    all_text = ""
    all_text_path = os.path.join(res_dir, "{}.txt".format('model_all_generated'))
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    program_dump_path = os.path.join(res_dir, "generated")
    
    uniqueness_file_name = output_path + "uniqueness_evaluation.txt"
    with open(str(uniqueness_file_name), "w") as stx_res_file:
        stx_res_file.write("\n" + "Uniqueness(count,%) and Generated Unseen(count,%)")
        
    # Load the vocabulary of the trained model
    dataset, vocab, n_bmpvectors = load_input_file(bitmap_file_path, vocabulary_path)
    tgt_start = vocab["tkn2idx"]["<s>"]
    tgt_end = vocab["tkn2idx"]["m)"]
    tgt_pad = vocab["tkn2idx"]["<pad>"]

    simulator = Simulator(vocab["idx2tkn"])

    bmpVector = dataset["bitmapVectors"]
    bmpVectorIndex = dataset["sources"]
    
    train_data, _, _ = load_input_file(train_file_path, vocabulary_path)
    train_codes = train_data["targets"]
    
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
    
    syntax_checker = PySyntaxChecker(vocab["tkn2idx"], use_cuda)
    if use_grammar:
        model.set_syntax_checker(syntax_checker)
        
    total_nb = 0
    total_failure = 0
    numb_unique = 0
    numb_unseen = 0
    
    unique_count_5 = 0
    unique_count_10 = 0
    #unique_count_50 = 0
    #unique_count_90 = 0
    
    unseen_count_5 = 0
    unseen_count_10 = 0
    #unseen_count_50 = 0
    #unseen_count_90 = 0
    
  #  dataset = shuffle_dataset(dataset, batch_size, randomize=False)
    
    saved_pred_all = []
    unique_pred_all = []
    unseen_pred_all = []
    failed_syntax_all = 0
    bitmap_mismatch_all = 0
    
    total_score = []
    total_unseen_score = []
    
    
    for sp_idx in tqdm(range(0, len(dataset["sources"]), batch_size)):
        
        tgt_inp_sequences,in_src_seq, tgt_seq_list, out_tgt_seq, srcs,targets,_  = get_minibatch(dataset, sp_idx, batch_size,
                                                tgt_start, tgt_end, tgt_pad)
        _,max_len = out_tgt_seq.size()
        if use_cuda:
            in_src_seq, out_tgt_seq = in_src_seq.cuda(), out_tgt_seq.cuda()
        
        syntax_failed_count = [0 for i in range(batch_size)]
        bitmap_mismatch_count = [0 for i in range(batch_size)]
        
        
        unique_pred = [[] for i in range(batch_size)]
        #quality_pred = [[] for i in range(batch_size)]
        corsp_domain = [[] for i in range(batch_size)]
        unseen_flag = [[] for i in range(batch_size)]
        target_pred = [[] for i in range(batch_size)]
        #failed_pred = [[] for i in range(batch_size)]
        unseen_pred = [[] for i in range(batch_size)]
        
        quality_zero_code = [3, 4, 20, 16, 17, 21]
        decoded = []
        for domain_K in range(0,n_domains):
            
            tgt_encoder_vector = torch.Tensor(len(in_src_seq), n_domains).fill_(0)
            
            
            index = torch.tensor(domain_K)
            if use_cuda:
                tgt_encoder_vector, index = tgt_encoder_vector.cuda(), index.cuda()
            
            tgt_encoder_vector.index_fill_(1, index, 1)
            decoded.append( model.beam_sample(in_src_seq, tgt_encoder_vector, tgt_start, tgt_end, 
                                        max_len,beam_size, top_k, domain_K)[0])
            
        temp = []
        for elem in decoded:
            for inner_elem in elem:
                temp.append(inner_elem)
        listOfTuples = list(temp)
        listOfTuples.sort(reverse=True)
        decoded = [[(ele) for ele in listOfTuples[:]]]

        for batch_idx, (target, sp_decoded) in \
            enumerate(zip(out_tgt_seq.chunk(out_tgt_seq.size(0)), decoded)):
            total_nb += 1 #should be batch size * number of IOs
            target = target.cpu().data.squeeze().numpy().tolist()
            target = [tkn_idx for tkn_idx in target if tkn_idx != tgt_pad]
            trgt_tkns = [vocab["idx2tkn"][tkn_idx] for tkn_idx in target]
            trgt_bmp_vec, _, _ = getBitmapVector(trgt_tkns)
            target_pred[batch_idx].append([bmpVector.index(trgt_bmp_vec), domain_K])
            
            if dump_programs:
                decoded_dump_dir = os.path.join(program_dump_path, str(str(sp_idx + batch_idx)))
                if not os.path.exists(decoded_dump_dir):
                    os.makedirs(decoded_dump_dir)
                write_program(os.path.join(decoded_dump_dir, "target"), target, vocab["idx2tkn"])
            
            selected_codes_count = 0
            # Correct syntaxes
            for rank, dec in enumerate(sp_decoded):
                #model_failed = True
                #model_quality_failed = True
                failed_syntax = False
                pred = dec[-2]
                ll = dec[0]
                domain_K = dec[-1]
                
                parse_success, cand_prog, cand_prog_json = simulator.get_prog_ast(pred)
                
                if parse_success:
                    pred_tkns = [vocab["idx2tkn"][tkn_idx] for tkn_idx in pred]
                    pred_bmp_vec, _, _ = getBitmapVector(pred_tkns)
                    ## Bitmap present in train set
                    if(pred_bmp_vec in bmpVector):
                        # Bitmap matches with target
                        if(bmpVector.index(trgt_bmp_vec) == bmpVector.index(pred_bmp_vec)):
                            #model_failed = False
                            #saved_pred_all.append(pred)
                            if(not(pred in unique_pred[batch_idx])):
                                unique_pred[batch_idx].append(pred)
                                corsp_domain[batch_idx].append(domain_K)
                                unique_pred_all.append(pred)
                                selected_codes_count+=1
                                if(not(pred in train_codes)):
                                    #model_quality_failed = False
                                    #quality_pred[batch_idx].append(pred)
                                    unseen_flag[batch_idx].append(True)
                                    unseen_pred[batch_idx].append(pred)
                                    unseen_pred_all.append(pred)
                                else:
                                    unseen_flag[batch_idx].append(False)
                                if dump_programs:
                                    file_name = str(domain_K)+ " - " + str(rank) + " - " + str(ll) 
                                    write_program(os.path.join(decoded_dump_dir, file_name), pred, vocab["idx2tkn"])
                else:
                    failed_syntax = True
                    
                #if(model_failed):
                    #failed_pred[batch_idx].append(pred)
                    #if(failed_syntax):
                        #syntax_failed_count[batch_idx] += 1
                        #failed_syntax_all += 1
                    #else:
                        #bitmap_mismatch_count[batch_idx] += 1
                        #bitmap_mismatch_all += 1
                #if(model_quality_failed):
                    #quality_pred[batch_idx].append(quality_zero_code)
                if(selected_codes_count == top_k):
                    break
                
        for i in range(0,batch_size):
            
            while(len(unique_pred[batch_idx]) < top_k):
                unique_pred[batch_idx].append(quality_zero_code)
                unseen_flag[batch_idx].append(False)
                corsp_domain[batch_idx].append(-1)
                total_failure += 1
                        
        for i in range(0,batch_size):
            indv_scores = []
            unseen_scores = []
            with open(str(uniqueness_file_name), "a") as stx_res_file:
                stx_res_file.write("\nBitmapVector " + str((sp_idx+1) + i) + " -> ")
                #numb_unique = (len(set(map(tuple, unique_pred[i]))))
                numb_unique = len(unique_pred[batch_idx])
                if(numb_unique> 4):
                    unique_count_5 += 1
                if(numb_unique> 9):
                    unique_count_10 += 1
                #if(numb_unique> 49):
                    #unique_count_50 += 1
                #if(numb_unique> 89):
                    #unique_count_90 += 1
                
                if(extra_info):
                    text += "TargetVectors: " + str(target_pred[i])  + "\n"
                    text += "Failed Syntax: " + str(syntax_failed_count[i])  + "\n"
                    text += "Bitmap Mismatch: " + str(bitmap_mismatch_count[i])  + "\n"
            
                    text += "Passed"  + "\n"
                for unique_prog, unseen_value, k_value in zip(unique_pred[i],unseen_flag[i],corsp_domain[i]):
                #for unique_prog in unique_pred[i] :
                    pred_tkns = [vocab["idx2tkn"][tkn_idx] for tkn_idx in unique_prog]
                    
                    if(eval_quality):
                        _, _,prg_ast_json = simulator.get_prog_ast(unique_prog)
                        code_json = iclr18_codejson_to_karelgym_codejson(prg_ast_json)
                        
                        
                        code = Code('karel', code_json)
                        score = obtain_karel_saturation_score_for_code(code, 200)
                        
                        indv_scores.append(score)
                        if(unseen_value == True):
                            unseen_scores.append(score)
                            total_unseen_score.append(score)
                        total_score.append(score)
                
                    if(extra_info):
                        text += str(k_value) + "    " + str(pred_tkns)  + "\n"
                    else:
                        text += str(pred_tkns)  + "\n"
                #if(extra_info):
                    #text += "Failed"  + "\n"
                    #for failed_prog in failed_pred[i]:
                        #pred_tkns = [vocab["idx2tkn"][tkn_idx] for tkn_idx in failed_prog]
                        #text += str(pred_tkns)  + "\n"
                        
                #for quality_prog in quality_pred[i]:
                    #pred_tkns = [vocab["idx2tkn"][tkn_idx] for tkn_idx in quality_prog]
                    #all_text += str(pred_tkns)  + "\n"
                
                stx_res_file.write("numb_unique : " + str(numb_unique)+ " , " )
                stx_res_file.write(str(100*numb_unique/ (n_domains )))
                
                numb_unseen = (len(set(map(tuple, unseen_pred[i]))))
                
                if(numb_unseen> 4):
                    unseen_count_5 += 1
                if(numb_unseen> 9):
                    unseen_count_10 += 1
                #if(numb_unseen> 49):
                    #unseen_count_50 += 1
                #if(numb_unseen> 89):
                    #unseen_count_90 += 1
                
                stx_res_file.write(";    numb_unseen : " + str(numb_unseen)+ " , " )
                stx_res_file.write(str(100*numb_unseen/ (n_domains )))
                
                stx_res_file.write(";    quality score : " + str(np.mean(indv_scores)))
                stx_res_file.write(";    quality unseen: " + str(np.mean(unseen_scores)))
    with open(str(uniqueness_file_name), "a") as stx_res_file:
        stx_res_file.write("\n" + "total unique : ")
        numb_unique = len(set(map(tuple, unique_pred_all)))
        stx_res_file.write(str(numb_unique)+ " , " )
        stx_res_file.write(str( (100*numb_unique) / ((len(dataset["sources"]))*n_domains ) ))
        
        stx_res_file.write("\n" + "total failure : ")
        stx_res_file.write(str(total_failure)+ " , " )
        stx_res_file.write(str( (100*total_failure) / ((len(dataset["sources"]))*n_domains ) ))
        
        numb_unseen = len(set(map(tuple, unseen_pred_all)))
        stx_res_file.write(";    total unseen : " + str(numb_unseen)+ " , " )
        stx_res_file.write(str( (100*numb_unseen) / ((len(dataset["sources"]))*n_domains ) ))
        stx_res_file.write(";    total score : " + str(np.mean(total_score))+ " , " )
        stx_res_file.write(";    total unseen score : " + str(np.mean(total_unseen_score))+ " , " )
        #numb_bmp = len(saved_pred_all)
        #stx_res_file.write("\n" + "total matching bmp vector : " + str(numb_bmp)+ " , " )
        #stx_res_file.write(str( (100*numb_bmp) / ((len(dataset["sources"]))*n_domains ) ))
        
        stx_res_file.write("\n" + "Unique 5 : " + str(unique_count_5)+ " , " )
        stx_res_file.write("\n" + "Unique 10 : " + str(unique_count_10)+ " , " )
        #stx_res_file.write("\n" + "Unique 50 : " + str(unique_count_50)+ " , " )
        #stx_res_file.write("\n" + "Unique 90 : " + str(unique_count_90)+ " , " )
        
        stx_res_file.write("\n" + "Unseen 5 : " + str(unseen_count_5)+ " , " )
        stx_res_file.write("\n" + "Unseen 10 : " + str(unseen_count_10)+ " , " )
        #stx_res_file.write("\n" + "Unseen 50 : " + str(unseen_count_50)+ " , " )
        #stx_res_file.write("\n" + "Unseen 90 : " + str(unseen_count_90)+ " , " )
        
        #stx_res_file.write("\n" + " failed syntax : " + str(failed_syntax_all)+ " , " )
        #stx_res_file.write("\n" + " bitmap mismatch : " + str(bitmap_mismatch_all)+ " , " )
    
    with open(text_path, 'w') as f:
        f.write(text)
    #with open(all_text_path, 'w') as f:
        #f.write(all_text)
    #uniquenss_value = (100*numb_unique) / ((len(dataset["sources"]))*n_domains ) 
    return np.mean(total_score)
    #return unseen_count_5

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
