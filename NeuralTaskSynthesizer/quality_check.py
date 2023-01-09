import time
import numpy as np
import argparse
import logging
import os 

from pathlib import Path
from karel.consistency import Simulator
from src.neural_code2task_syn.task_synthesizer import obtain_karel_saturation_score_for_code

from src.karel_data_converters.converter_format_iclr18_to_karelgym import \
    iclr18_codejson_to_karelgym_codejson

from src.karel_emulator.code import Code

def add_args(parser):
    
    parse_group = parser.add_argument_group("Quality Check",
                                        description="Quality check options")
    
    
    parse_group.add_argument("--input_code_file", type=str,
                            default="code_val.txt",
                            help="Path to the input code file. "
                            " Default: %(default)s")
    parse_group.add_argument("--result_folder", type=str,
                            default="exps",
                            help="Where to store the results. "
                            " Default: %(default)s")

    parse_group.add_argument("--vocab", type=str,
                            default="data/new_vocab.vocab",
                            help="Path to the output vocabulary."
                            " Default: %(default)s")
    parse_group.add_argument('--n_domains', type=int,
                             default=20,
                             help="Number of domains for target encoder. "
                             "Default: %(default)s")
    parse_group.add_argument('--top_k', type=int,
                             default=1,
                             help="Number of codes to consider per code sketch. "
                             "Default: %(default)s")
    

def translate(seq,
              vocab):
    return [vocab[str(elt)] for elt in seq]

def return_sim(path_to_vocab):
    tgt_tkn2idx = {
        '<pad>': 0,
    }
    next_id = 1
    with open(path_to_vocab, 'r') as vocab_file:
        for line in vocab_file.readlines():
            tgt_tkn2idx[line.strip()] = next_id
            next_id += 1
    tgt_idx2tkn = {}
    for tkn, idx in tgt_tkn2idx.items():
        tgt_idx2tkn[idx] = tkn

    vocab = {"idx2tkn": tgt_idx2tkn,
                "tkn2idx": tgt_tkn2idx}
        
    simulator = Simulator(vocab["idx2tkn"])
    return simulator,tgt_tkn2idx

#program_json = {"run": [{"body": [{"body": [{"body": [{"type": "move"},
                                                        #{"condition": {
                                                            #"type": "rightIsClear"},
                                                            #"elseBody": [
                                                                #{
                                                                    #"type": "turnLeft"}],
                                                            #"ifBody": [
                                                                #{
                                                                    #"type": "putMarker"}],
                                                            #"type": "ifElse"}],
                                                #"times": 3, "type": "repeat"}],
                                    #"condition": {"type": "frontIsClear"},
                                    #"type": "if"}], "times": 3, "type": "repeat"},
                            #{"type":
                                #"putMarker"},
                            #{"body": [{"type": "turnRight"}], "times": 2,
                            #"type": "repeat"}, {"type": "move"}]}
                            
def Sort(sub_li):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][1] < sub_li[j + 1][1]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li

if __name__ == '__main__':
    #Creating Parser
    parser = argparse.ArgumentParser(
        description='Convert code to code types.')
    add_args(parser)
    args = parser.parse_args()
    
    # Creating the results directory
    result_dir = Path(args.result_folder)
    if not result_dir.exists():
        os.makedirs(str(result_dir))
        
    # Setting up the logs
    log_file = result_dir /(str(result_dir) + "_logs.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=str(log_file),
        filemode='w'
    )

        
    #Creating Simulator
    simulator,tgt_tkn2idx = return_sim(args.vocab)
    top_k_score = []
    top_1_score = []
    total_score = []

    #Parsing code file
    code_file = open(args.input_code_file, 'r')
    Lines = code_file.readlines() 
    code_index = 1
    index = 0
    sorted_codes = []
    
    text = ""
    text_path = os.path.join(result_dir, "{}.txt".format('final_selected'))
    
    for line in Lines:
        index += 1
        prog_extracted= []
        prog = line.split(" ")
        for token in prog:
            token = token.replace("'", "")
            token = token.replace('"', '')
            token = token.replace(' ', '')
            token = token.replace(',', '')
            token = token.replace('[', '')
            token = token.replace(']', '')
            token = token.replace('\n', '')
            prog_extracted.append(token)
            
        prog_extracted_idces = translate(prog_extracted, tgt_tkn2idx)
        _, _,prg_ast_json = simulator.get_prog_ast(prog_extracted_idces)
        code_json = iclr18_codejson_to_karelgym_codejson(prg_ast_json)

        code = Code('karel', code_json)
        scores = []
        start = time.time()
        for _ in range(1):
            score = obtain_karel_saturation_score_for_code(code, 2000)
            scores.append(score)
        end = time.time()
    #  print(f"Time taken: {end - start}")

        logging.info("Score for code %d. :: %.5f." % (code_index, np.mean(scores)))
        code_index += 1
    #  print(np.std(scores))
        sorted_codes.append([prog,np.mean(scores)])
        
        total_score.append(np.mean(scores))
        logging.info("total_score :: %.5f." % np.mean(total_score))
        
        if(index == args.n_domains):
            index = 0
            temp = (Sort(sorted_codes)[0:args.top_k])
            #sorted_codes = []
            for code_score in temp:
                top_k_score.append(code_score[1])
                text += str(code_score[0])  + "\n"
                logging.info("code selected :: %.5f." % code_score[1])
                logging.info("top_k_score :: %.5f." % np.mean(top_k_score))
                
            temp = (Sort(sorted_codes)[0:1])
            sorted_codes = []
            for code_score in temp:
                top_1_score.append(code_score[1])
                #text += str(code_score[0])  + "\n"
                logging.info("code selected :: %.5f." % code_score[1])
                logging.info("top_1_score :: %.5f." % np.mean(top_1_score))

    with open(text_path, 'w') as f:
            f.write(text)
