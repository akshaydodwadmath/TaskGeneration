from __future__ import division
# External imports
import json
import os
import torch

from torch.autograd import Variable
from tqdm import tqdm

from dataloader import load_input_file, get_minibatch, shuffle_dataset
from karel.consistency import Simulator
from preprocessing.code_to_codeType import getFeatureVector

def add_eval_args(parser):
    parser.add_argument('--use_grammar', action="store_true")
    parser.add_argument("--nb_samples", type=int,
                        default=0,
                        help="How many samples to use to compute the accuracy."
                        "Default: %(default)s, for all the dataset")
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
                   feature_file_path,
                   train_file_path,
                   nb_samples,
                   n_domains,
                   use_grammar,
                   output_path,
                   beam_size,
                   top_k,
                   batch_size,
                   use_cuda,
                   dump_programs):
    
    all_semantic_output_path = []
    all_syntax_output_path = []
    all_featVec_present_output_path = []
    all_featVec_match_output_path = []
    res_dir = os.path.dirname(output_path)
    
    text = ""
    text_path = os.path.join(res_dir, "{}.txt".format('model_generated'))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for k in range(top_k):
        
        new_semantic_term = "semantic_top%d.txt" % (k+1)
        new_semantic_file_name = output_path + new_semantic_term
        
        new_syntax_term = "syntax_top%d.txt" % (k+1)
        new_syntax_file_name = output_path + new_syntax_term
        
        new_featVec_present_term = "featVec_present_top%d.txt" % (k+1)
        new_featVec_present_file_name = output_path + new_featVec_present_term
        
        new_featVec_match_term = "featVec_match_top%d.txt" % (k+1)
        new_featVec_match_file_name = output_path + new_featVec_match_term
        

        all_semantic_output_path.append(new_semantic_file_name)
        all_syntax_output_path.append(new_syntax_file_name)
        all_featVec_present_output_path.append(new_featVec_present_file_name)
        all_featVec_match_output_path.append(new_featVec_match_file_name)
        
    program_dump_path = os.path.join(res_dir, "generated")
    
    uniqueness_file_name = output_path + "uniqueness_evaluation.txt"
    with open(str(uniqueness_file_name), "w") as stx_res_file:
        stx_res_file.write("\n" + "Uniqueness(count,%) and Generated Unseen(count,%)")
        
    # Load the vocabulary of the trained model
    dataset, vocab, nfeaturevectors = load_input_file(feature_file_path, vocabulary_path)
    tgt_start = vocab["tkn2idx"]["<s>"]
    tgt_end = vocab["tkn2idx"]["m)"]
    tgt_pad = vocab["tkn2idx"]["<pad>"]

    simulator = Simulator(vocab["idx2tkn"])

    fVector = dataset["featureVectors"]
    fVectorIndex = dataset["sources"]
    
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
    
    if use_grammar:
        syntax_checker = PySyntaxChecker(vocab["tkn2idx"], use_cuda)
        model.set_syntax_checker(syntax_checker)
        
        
    nb_semantic_correct = [0 for _ in range(top_k)]
    nb_syntax_correct = [0 for _ in range(top_k)]
    nb_featVec_present = [0 for _ in range(top_k)]
    nb_featVec_match = [0 for _ in range(top_k)]
    total_nb = 0
    numb_unique = 0
    numb_unseen = 0
    
    unique_count_5 = 0
    unique_count_10 = 0
    unique_count_50 = 0
    unique_count_90 = 0
    
    unseen_count_5 = 0
    unseen_count_10 = 0
    unseen_count_50 = 0
    unseen_count_90 = 0
    
  #  dataset = shuffle_dataset(dataset, batch_size, randomize=False)
    
    saved_pred_all = []
    unique_pred_all = []
    unseen_pred_all = []
    failed_syntax_all = 0
    feature_mismatch_all = 0
    
    
    for sp_idx in tqdm(range(0, len(dataset["sources"]), batch_size)):
        
        tgt_inp_sequences,in_src_seq, out_tgt_seq, srcs,targets,  = get_minibatch(dataset, sp_idx, batch_size,
                                                tgt_start, tgt_end, tgt_pad)
        _,max_len = out_tgt_seq.size()
        if use_cuda:
            in_src_seq, out_tgt_seq = in_src_seq.cuda(), out_tgt_seq.cuda()
        
        sample_count = [0 for i in range(batch_size)]
        syntax_failed_count = [0 for i in range(batch_size)]
        feat_mismatch_count = [0 for i in range(batch_size)]
        
        
        unique_pred = [[] for i in range(batch_size)]
        corsp_domain = [[] for i in range(batch_size)]
        target_pred = [[] for i in range(batch_size)]
        failed_pred = [[] for i in range(batch_size)]
        unseen_pred = [[] for i in range(batch_size)]
        for K in range(0,n_domains):
            
            tgt_encoder_vector = torch.Tensor(len(in_src_seq), n_domains).fill_(0)
            
            
            index = torch.tensor(K)
            if use_cuda:
                tgt_encoder_vector, index = tgt_encoder_vector.cuda(), index.cuda()
            
            tgt_encoder_vector.index_fill_(1, index, 1)
            decoded = model.beam_sample(in_src_seq, tgt_encoder_vector, tgt_start, tgt_end, 
                                        max_len,beam_size, top_k)
            for batch_idx, (target, sp_decoded) in \
                enumerate(zip(out_tgt_seq.chunk(out_tgt_seq.size(0)), decoded)):
                
                total_nb += 1 #should be batch size * number of IOs
                target = target.cpu().data.squeeze().numpy().tolist()
                target = [tkn_idx for tkn_idx in target if tkn_idx != tgt_pad]
                trgt_tkns = [vocab["idx2tkn"][tkn_idx] for tkn_idx in target]
                trgt_feat_vec, _ = getFeatureVector(trgt_tkns)
                target_pred[batch_idx].append([fVector.index(trgt_feat_vec), K])
                
                if dump_programs:
                    decoded_dump_dir = os.path.join(program_dump_path, str(str(sp_idx + batch_idx)))
                    if not os.path.exists(decoded_dump_dir):
                        os.makedirs(decoded_dump_dir)
                    write_program(os.path.join(decoded_dump_dir, "target"), target, vocab["idx2tkn"])
                
                ranked_entered = False
                
                # Correct syntaxes
                for rank, dec in enumerate(sp_decoded):
                    model_failed = True
                    if(sample_count[batch_idx]< nb_samples):
                        sample_count[batch_idx] += 1
                        failed_syntax = False
                        pred = dec[-1]
                        ll = dec[0]
                        parse_success, cand_prog = simulator.get_prog_ast(pred)
                        if parse_success:
                            pred_tkns = [vocab["idx2tkn"][tkn_idx] for tkn_idx in pred]
                            pred_feat_vec, _ = getFeatureVector(pred_tkns)
                            ## Feature present in train set
                            if(pred_feat_vec in fVector):
                                # Feature matches with target
                                if(fVector.index(trgt_feat_vec) == fVector.index(pred_feat_vec)):
                                    model_failed = False
                                    saved_pred_all.append(pred)
                                    if(not(pred in unique_pred[batch_idx])):
                                        unique_pred[batch_idx].append(pred)
                                        corsp_domain[batch_idx].append(K)
                                        unique_pred_all.append(pred)
                                        
                                        if(not(pred in train_codes)):
                                            unseen_pred[batch_idx].append(pred)
                                            unseen_pred_all.append(pred)
                                        if dump_programs:
                                            file_name = str(K)+ " - " + str(rank) + " - " + str(ll) 
                                            write_program(os.path.join(decoded_dump_dir, file_name), pred, vocab["idx2tkn"])
                                    if(not ranked_entered):
                                        for top_idx in range(rank, top_k):
                                            nb_featVec_match[top_idx] += 1
                                        ranked_entered = True
                        else:
                            failed_syntax = True
                            
                        if(model_failed):
                            failed_pred[batch_idx].append(pred)
                            if(failed_syntax):
                                syntax_failed_count[batch_idx] += 1
                                failed_syntax_all += 1
                            else:
                                feat_mismatch_count[batch_idx] += 1
                                feature_mismatch_all += 1
                            
        for i in range(0,batch_size):
            with open(str(uniqueness_file_name), "a") as stx_res_file:
                stx_res_file.write("\nFeatureVector " + str((sp_idx+1) + i) + " -> ")
                numb_unique = (len(set(map(tuple, unique_pred[i]))))
                
                if(numb_unique> 4):
                    unique_count_5 += 1
                if(numb_unique> 9):
                    unique_count_10 += 1
                if(numb_unique> 49):
                    unique_count_50 += 1
                if(numb_unique> 89):
                    unique_count_90 += 1
                
                text += "TargetVectors: " + str(target_pred[i])  + "\n"
                text += "Failed Syntax: " + str(syntax_failed_count[i])  + "\n"
                text += "Feature Mismatch: " + str(feat_mismatch_count[i])  + "\n"
                
                
                text += "Passed"  + "\n"
                for unique_prog,k_value in zip(unique_pred[i],corsp_domain[i]):
                    pred_tkns = [vocab["idx2tkn"][tkn_idx] for tkn_idx in unique_prog]
                    text += str(k_value) + "    " + str(pred_tkns)  + "\n"
                
                text += "Failed"  + "\n"
                for failed_prog in failed_pred[i]:
                    pred_tkns = [vocab["idx2tkn"][tkn_idx] for tkn_idx in failed_prog]
                    text += str(pred_tkns)  + "\n"
                
                stx_res_file.write("numb_unique : " + str(numb_unique)+ " , " )
                stx_res_file.write(str(100*numb_unique/ (nb_samples )))
                
                numb_unseen = (len(set(map(tuple, unseen_pred[i]))))
                
                if(numb_unseen> 4):
                    unseen_count_5 += 1
                if(numb_unseen> 9):
                    unseen_count_10 += 1
                if(numb_unseen> 49):
                    unseen_count_50 += 1
                if(numb_unseen> 89):
                    unseen_count_90 += 1
                
                stx_res_file.write(";    numb_unseen : " + str(numb_unseen)+ " , " )
                stx_res_file.write(str(100*numb_unseen/ (nb_samples )))
                
               
            
        
    #for k in range(top_k):
        #with open(str(all_syntax_output_path[k]), "w") as stx_res_file:
            #stx_res_file.write(str(100*nb_syntax_correct[k]/total_nb))
            
    #for k in range(top_k):
        #with open(str(all_featVec_present_output_path[k]), "w") as stx_res_file:
            #stx_res_file.write(str(100*nb_featVec_present[k]/total_nb))
    with open(str(uniqueness_file_name), "a") as stx_res_file:
        stx_res_file.write("\n" + "total unique : ")
        numb_unique = len(set(map(tuple, unique_pred_all)))
        stx_res_file.write(str(numb_unique)+ " , " )
        stx_res_file.write(str( (100*numb_unique) / ((len(dataset["sources"]))*nb_samples ) ))
        
        numb_unseen = len(set(map(tuple, unseen_pred_all)))
        stx_res_file.write(";    total unseen : " + str(numb_unseen)+ " , " )
        stx_res_file.write(str( (100*numb_unseen) / ((len(dataset["sources"]))*nb_samples ) ))
        
        numb_feat = len(saved_pred_all)
        stx_res_file.write("\n" + "total matching feat vector : " + str(numb_feat)+ " , " )
        stx_res_file.write(str( (100*numb_feat) / ((len(dataset["sources"]))*nb_samples ) ))
        
        stx_res_file.write("\n" + "Unique 5 : " + str(unique_count_5)+ " , " )
        stx_res_file.write("\n" + "Unique 10 : " + str(unique_count_10)+ " , " )
        stx_res_file.write("\n" + "Unique 50 : " + str(unique_count_50)+ " , " )
        stx_res_file.write("\n" + "Unique 90 : " + str(unique_count_90)+ " , " )
        
        stx_res_file.write("\n" + "Unseen 5 : " + str(unseen_count_5)+ " , " )
        stx_res_file.write("\n" + "Unseen 10 : " + str(unseen_count_10)+ " , " )
        stx_res_file.write("\n" + "Unseen 50 : " + str(unseen_count_50)+ " , " )
        stx_res_file.write("\n" + "Unseen 90 : " + str(unseen_count_90)+ " , " )
        
        stx_res_file.write("\n" + " failed syntax : " + str(failed_syntax_all)+ " , " )
        stx_res_file.write("\n" + " feature mismatch : " + str(feature_mismatch_all)+ " , " )
    
    for k in range(top_k):
        with open(str(all_featVec_match_output_path[k]), "w") as stx_res_file:
            stx_res_file.write(str(100*nb_featVec_match[k]/total_nb))
    with open(text_path, 'w') as f:
        f.write(text)
    uniquenss_value = (100*numb_unseen) / ((len(dataset["sources"]))*nb_samples ) 
    return uniquenss_value
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
