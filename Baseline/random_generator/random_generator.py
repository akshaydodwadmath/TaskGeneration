import time
import numpy as np
import math
import argparse
import json
import random
import os

from pathlib import Path
from karel.consistency import Simulator
from itertools import product

from src.neural_code2task_syn.task_synthesizer import obtain_karel_saturation_score_for_code
from src.karel_data_converters.converter_format_iclr18_to_karelgym import \
    iclr18_codejson_to_karelgym_codejson
from src.karel_emulator.code import Code

actions = [
    'move',
    'turnLeft',
    'turnRight',
    'pickMarker',
    'putMarker',
]

commands = [
            ['REPEAT','r(','r)', 0],
            ['WHILE','w(','w)', 1],
            ['IF','i(','i)', 1]
            ]
command_if_else = [
            ['IFELSE','i(','i)', 1],
            ['ELSE','e(','e)', 1]
            ]

cond = [
    ['c(','markersPresent','c)'],
    ['c(','noMarkersPresent','c)'],
    ['c(','leftIsClear','c)'],
    ['c(','rightIsClear','c)'],
    ['c(','frontIsClear','c)'],
    ['c(','not','c(','leftIsClear','c)','c)'],
    ['c(','not','c(','rightIsClear','c)','c)'],
    ['c(','not','c(','frontIsClear','c)','c)'],
    ['R=']# repeat options 2 to 10
    ]

control_close = ['m)',
         'r)',
         'w)',
         'i)',
         'e)',
    ]

n_action_C1 = [2,11, 12]

n_action_C2_R = [5,10,15]

n_action_C3_RR = [4,9,14]
n_action_C3_RW = [6,11]

n_action_C4_RRR = [3,6,11]
n_action_C4_RWR = [7,12]
n_action_C4_RRW = [3,8,13]
n_action_C4_RWW = [4,9,14]

n_action_C5_RW = [2,7,12]
n_action_C5_RR = [5,10,15]
n_action_C5_RIE = [6,11]
n_action_C5_RI = [2,7,12]

n_action_C6_RRR = [3,8,13]
n_action_C6_RWR = [4,9,14]
n_action_C6_RRW = [5,10,15]
n_action_C6_RWW = [6,11]
n_action_C6_RIEW = [5,10,15]
n_action_C6_RIER = [7,8,13]
n_action_C6_RIW = [3,8,13]
n_action_C6_RIR = [6,11]

n_action_C7_RRR = [4,9,14]
n_action_C7_RWR = [5,10,15]
n_action_C7_RRW = [6,11]
n_action_C7_RWW = [7,12]
n_action_C7_RRIE = [5,10,15]
n_action_C7_RWIE = [7,12]
n_action_C7_RWI = [5,10,15]
n_action_C7_RRI = [3,3,8,8,13]
n_action_C7_WWR = [5]

n_action_C8_RRR = [5,10,15]
n_action_C8_RWR = [6,11] 
n_action_C8_RRW = [7,12]
n_action_C8_RWW = [3,8,13]
n_action_C8_RIEW = [4,9,14]
n_action_C8_RIER = [7,12,13]
n_action_C8_RRIE = [5,10,15]
n_action_C8_RWIE = [7,12]
n_action_C8_RIEIE = [8,13]
n_action_C8_RII = [3,4,5,6,7,8,11,12,13,14,15]
n_action_C8_RIEI = [8,13]
n_action_C8_RIIE = [6,11]
n_action_C8_WIR = [12]

n_action_list = [n_action_C1, n_action_C2_R, n_action_C3_RR, n_action_C3_RW,
                n_action_C4_RRR, n_action_C4_RWR, n_action_C4_RRW, n_action_C4_RWW,
                n_action_C5_RW, n_action_C5_RR, n_action_C5_RIE, n_action_C5_RI,
                
                n_action_C6_RRR, n_action_C6_RWR, n_action_C6_RRW, n_action_C6_RWW, 
                n_action_C6_RIEW,  n_action_C6_RIER, n_action_C6_RIW, n_action_C6_RIR,
                
                n_action_C7_RRR, n_action_C7_RWR, n_action_C7_RRW, n_action_C7_RWW,
                n_action_C7_RRIE, n_action_C7_RWIE, n_action_C7_RWI, n_action_C7_RRI, n_action_C7_WWR,
                
                n_action_C8_RRR, n_action_C8_RWR, n_action_C8_RRW, n_action_C8_RWW,
                n_action_C8_RIEW, n_action_C8_RIER, n_action_C8_RRIE, n_action_C8_RWIE,
                n_action_C8_RIEIE, n_action_C8_RII, n_action_C8_RIEI, n_action_C8_RIIE, n_action_C8_WIR
    ]
required_ctypes =   [ 
                    ['action'], #Type01 CT1
                   
                    ['action', 
                        'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1', 
                        'action' ], #Type02 #CT2
                
                    ['action', 
                        'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1',
                        'action', 
                        'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                        'action'],#Type04 #CT3
                    ['action', 
                        'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1',
                        'action', 
                        'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                        'action'],#Type04 #CT3
                    
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1',
                     'action', 
                     'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action',
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action'],#Type42 #CT4
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1',
                     'action', 
                     'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action',
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action'],#Type42 #CT4
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1',
                     'action', 
                     'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action',
                     'WHILE', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action'],#Type42 #CT4
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1',
                     'action', 
                     'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action',
                     'WHILE', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action'],#Type42 #CT4
                   
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action', 'cclose1', 
                     'action'], #Type 08 #CT5
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action', 'cclose1', 
                     'action'], #Type 08 #CT5
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action', 'cclose1',
                     'action'],#Type 09 #CT5
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'IF', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action', 'cclose1', 
                     'action'], #Type 08 #CT5
                    
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action', 'cclose1', 
                     'action',
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action'], #Type 18 #CT6
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action', 'cclose1', 
                     'action',
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action'], #Type 18 #CT6
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action', 'cclose1', 
                     'action',
                     'WHILE', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action'], #Type 18 #CT6
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action', 'cclose1', 
                     'action',
                     'WHILE', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action'], #Type 18 #CT6
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action', 'cclose1',
                     'action',
                     'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action'],#Type 19 #CT6
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action', 'cclose1',
                     'action',
                     'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action'],#Type 19 #CT6
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'IF', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action', 'cclose1', 
                     'action',
                     'WHILE', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action'], #Type 18 #CT6
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'action',
                     'IF', 'cond2', 'copen2', 'mandate_action', 'cclose2', 
                     'action', 'cclose1', 
                     'action',
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action'], #Type 18 #CT6
                    
                    ['action',
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1', 
                     'action',
                     'REPEAT', 'cond2', 'copen2', 'action',
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose2', 
                     'action'], #Type 30 #CT7
                    ['action',
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1', 
                     'action',
                     'WHILE', 'cond2', 'copen2', 'action',
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose2', 
                     'action'], #Type 30 #CT7
                    ['action',
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1', 
                     'action',
                     'REPEAT', 'cond2', 'copen2', 'action',
                     'WHILE', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose2', 
                     'action'], #Type 30 #CT7
                    ['action',
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1', 
                     'action',
                     'WHILE', 'cond2', 'copen2', 'action',
                     'WHILE', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose2', 
                     'action'], #Type 30 #CT7
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1', 
                     'action',
                     'REPEAT', 'cond2', 'copen2', 'action',
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action', 'cclose2',
                     'action'],#Type 31 #CT7
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1', 
                     'action',
                     'WHILE', 'cond2', 'copen2', 'action',
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action', 'cclose2',
                     'action'],#Type 31 #CT7
                    ['action',
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1', 
                     'action',
                     'WHILE', 'cond2', 'copen2', 'action',
                     'IF', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose2', 
                     'action'], #Type 30 #CT7
                    ['action',
                     'REPEAT', 'cond1', 'copen1', 'mandate_action', 'cclose1', 
                     'action',
                     'REPEAT', 'cond2', 'copen2', 'action',
                     'IF', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose2', 
                     'action'], #Type 30 #CT7
                    ['action',
                     'WHILE', 'cond1', 'copen1', 'mandate_action', 'cclose1', 
                     'action',
                     'WHILE', 'cond2', 'copen2', 'action',
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose2', 
                     'action'], #Type 30 #CT7
                
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose1',
                     'action'],#Type50 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose1',
                     'action'],#Type50 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 
                     'WHILE', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose1',
                     'action'],#Type50 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 
                     'WHILE', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose1',
                     'action'],#Type50 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action',
                     'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 'cclose1',
                     'action'],#Type51 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action',
                     'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 'cclose1',
                     'action'],#Type51 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'REPEAT', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action', 'cclose1',
                     'action'],#Type52 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'WHILE', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action', 'cclose1',
                     'action'],#Type52 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose',  
                     'action',
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action', 'cclose1',
                     'action'],#Type53 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'IF', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 
                     'IF', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose1',
                     'action'],#Type50 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action',
                     'IF', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 'cclose1',
                     'action'],#Type51 #CT8
                    ['action', 
                     'REPEAT', 'cond1', 'copen1', 
                     'action', 
                     'IF', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 
                     'cif', 'c_cndif', 'c_ifopen', 'mandate_action', 'c_ifclose',
                     'celse', 'c_elseopen', 'mandate_action', 'c_elseclose', 
                     'action', 'cclose1',
                     'action'],#Type52 #CT8
                    ['action', 
                     'WHILE', 'cond1', 'copen1', 
                     'action', 
                     'IF', 'cond2', 'copen2', 'mandate_action', 'cclose2',
                     'action', 
                     'REPEAT', 'cond3', 'copen3', 'mandate_action', 'cclose3', 
                     'action', 'cclose1',
                     'action'],#Type50 #CT8
                    
                    ]
def add_args(parser):
    parse_group = parser.add_argument_group("Conversion",
                                        description="Conversion options")
    parse_group.add_argument('--data_dir', type=str, default='data')
    parse_group.add_argument("--data_generator", action="store_true")
    parse_group.add_argument("--num_codes_per_spec", type=int,
                             default=15,
                             help="Number of codes to generate per input specification")
    parse_group.add_argument("--min_index", type=int,
                             default=0,
                             help="required code types min index")
    parse_group.add_argument("--max_index", type=int,
                             default=0,
                             help="required code types max index")
    
    parse_group.add_argument('--quality_threshold', type=float,
                             default=0.5,
                             help="Threshold for code quality. "
                             "Default: %(default)s")
    parse_group.add_argument("--max_iterations", type=int,
                             default=200,
                             help="Max iterations for corresponding task generation")
    
    parse_group.add_argument("--vocab", type=str,
                            default="data/new_vocab.vocab",
                            help="Path to the output vocabulary."
                            " Default: %(default)s")
    
    
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

def generateCodes(code_type, selected_ctrl, max_nb_actions):
    code= []
    code += token_beg
    action_set = random.choices(actions, k=max_nb_actions)
    action_in_code_type = []
    ctrl_count = 0
    cifelse_count = 0
    for token in code_type:
        code.append(token)
    

    #Add actions
    while(action_set):
        index = 0
        for token in code:
            if(not action_set):
                break
            if('mandate_action' in token):
                code[index] = action_set.pop()
            index+=1
           
        index = 0
        for token in code:
            if(not action_set):
                break
            if(np.random.choice(2, 1) == 1): #prob = 0.5
                if('action' in token):
                    code[index] = action_set.pop()
                elif(token in actions):
                    code.insert(index+1, action_set.pop())
                
            
            index+=1
    
    #Add control statements and brackets
    open_set = []
    close_set = []
    cond_set = []
    index = 0
    for token in code:
        if('REPEAT' in token):
            #current_ctrl_set = selected_ctrl.pop()
            current_ctrl_set = commands[0]
            code[index] = current_ctrl_set[0]
            open_set.append(current_ctrl_set[1])
            close_set.append(current_ctrl_set[2])
            cond_set.append(current_ctrl_set[3])
        elif('WHILE' in token):
            #current_ctrl_set = selected_ctrl.pop()
            current_ctrl_set = commands[1]
            code[index] = current_ctrl_set[0]
            open_set.append(current_ctrl_set[1])
            close_set.append(current_ctrl_set[2])
            cond_set.append(current_ctrl_set[3])
        elif('IF' in token):
            #current_ctrl_set = selected_ctrl.pop()
            current_ctrl_set = commands[2]
            code[index] = current_ctrl_set[0]
            open_set.append(current_ctrl_set[1])
            close_set.append(current_ctrl_set[2])
            cond_set.append(current_ctrl_set[3])
        elif('copen' in token):
            code[index] = open_set[int(token[-1])-1]
        elif('cond' in token):
            if(cond_set[int(token[-1])-1] == 0):
                code[index] = [str(cond[-1][0]) + str(random.choice(max_repeat))][0]
            else:
                temp = random.choice(cond[:-1])
                for i in reversed(temp):
                    code.insert(index+1,i)
        elif('cclose' in token):
            code[index] = close_set[int(token[-1])-1]
        index+=1

    #Add if else statements
    index = 0
    for token in code:
        if('cif' in token):
            code[index]  = command_if_else[0][0]
            
        elif('c_cndif' in token):
            temp = random.choice(cond[:-1])
            for i in reversed(temp):
                code.insert(index+1,i)
        elif('celse' in token):
            code[index]  = command_if_else[1][0]
        elif('c_ifopen' in token):
            code[index]  = command_if_else[0][1]
        elif('c_ifclose' in token):
            code[index]  = command_if_else[0][2]
        elif('c_elseopen' in token):
            code[index]  = command_if_else[1][1]
        elif('c_elseclose' in token):
            code[index]  = command_if_else[1][2]
        index+=1
    
    #Remove redundant tokens
    index = 0
    ret_code = []
    while(index != len(code)):
        if (not(('action' in code[index]) or ('cond' in code[index]) or ('c_cndif' in code[index]))):
             ret_code.append(code[index])
        index +=1

    ret_code += token_end
    ret_code = list(filter(None, ret_code))
    return ret_code

def checkfor_pre_REPEAT_1(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if((token == 'REPEAT') and (gen_code[index+4]== 'r)')):
            if(gen_code[index-1] == gen_code[index+3]):
                return True
        index+=1
    return False

def checkfor_pre_REPEAT_2(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if((token == 'REPEAT')and (gen_code[index+5]== 'r)')):
            if((gen_code[index-2] == gen_code[index+3]) and 
                (gen_code[index-1] == gen_code[index+4])):
                return True
        index+=1
    return False

#def checkfor_pre_REPEAT_3(gen_code):
    #index = 1
  ##  print("gen_code", gen_code)
    #for token in gen_code[1:-1]:
        #if((token == 'REPEAT')and (gen_code[index+6]== 'r)')):
            #if((gen_code[index-3] == gen_code[index+3]) and 
                #(gen_code[index-2] == gen_code[index+4]) and 
                #(gen_code[index-1] == gen_code[index+5]) ):
                #return True
        #index+=1
    #return False

def checkfor_post_REPEAT_1(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if((token == 'r)') and (gen_code[index-2] == 'r(')):
            if(gen_code[index-1] == gen_code[index+1]):
                return True
        index+=1
    return False

def checkfor_post_REPEAT_2(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if((token == 'r)') and (gen_code[index-3] == 'r(')):
            if((gen_code[index-2] == gen_code[index+1]) and 
                (gen_code[index-1] == gen_code[index+2])):
                return True
        index+=1
    return False

#def checkfor_post_REPEAT_3(gen_code):
    #index = 1
  ##  print("gen_code", gen_code)
    #for token in gen_code[1:-1]:
        #if((token == 'r)')and (gen_code[index-4] == 'r(')):
            #if((gen_code[index-3] == gen_code[index+1]) and 
                #(gen_code[index-2] == gen_code[index+2]) and 
                #(gen_code[index-1] == gen_code[index+3]) ):
                #return True
        #index+=1
    #return False

def checkfor_notfrontIsClear_move(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if((token == 'not') and (gen_code[index+2] == 'frontIsClear')):
            if((gen_code[index+6] == 'move')):
                return True
        index+=1
    return False

def checkfor_notrightIsClear_tR(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if((token == 'not') and (gen_code[index+2] == 'rightIsClear')):
            if((gen_code[index+6] == 'turnRight')):
                return True
        index+=1
    return False

def checkfor_notleftIsClear_tL(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if((token == 'not') and (gen_code[index+2] == 'leftIsClear')):
            if((gen_code[index+6] == 'turnLeft')):
                return True
        index+=1
    return False

def checkfor_frontIsClear_tL_tR(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'frontIsClear'):
            if((gen_code[index+3] == 'turnLeft') or (gen_code[index+3] == 'turnRight')):
                return True
        index+=1
    return False

def checkfor_rightIsClear_tL(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'rightIsClear'):
            if((gen_code[index+3] == 'turnLeft')):
                return True
        index+=1
    return False

def checkfor_leftIsClear_tR(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'leftIsClear'):
            if((gen_code[index+3] == 'turnRight')):
                return True
        index+=1
    return False

def checkfor_nomarkerPresent_pickMarker(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'noMarkersPresent'):
            if((gen_code[index+3] == 'pickMarker')):
                return True
        index+=1
    return False

def checkfor_tL_tR(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'turnLeft'):
            if((gen_code[index-1] == 'turnRight') or (gen_code[index+1] == 'turnRight')):
                return True
        index+=1
    return False
        
def checkfor_tL_tL_tL(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'turnLeft'):
            if((gen_code[index-1] == 'turnLeft') and (gen_code[index+1] == 'turnLeft')):
                return True
        index+=1
    return False

def checkfor_tR_tR_tR(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'turnRight'):
            if((gen_code[index-1] == 'turnRight') and (gen_code[index+1] == 'turnRight')):
                return True
        index+=1
    return False

def checkfor_tL_pickM_tR(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'pickMarker'):
            if((gen_code[index-1] == 'turnLeft') and (gen_code[index+1] == 'turnRight')):
                return True
        index+=1
    return False

def checkfor_tR_pickM_tL(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'pickMarker'):
            if((gen_code[index-1] == 'turnRight') and (gen_code[index+1] == 'turnLeft')):
                return True
        index+=1
    return False

def checkfor_tL_putM_tR(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'putMarker'):
            if((gen_code[index-1] == 'turnLeft') and (gen_code[index+1] == 'turnRight')):
                return True
        index+=1
    return False

def checkfor_tR_putM_tL(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'putMarker'):
            if((gen_code[index-1] == 'turnRight') and (gen_code[index+1] == 'turnLeft')):
                return True
        index+=1
    return False

def checkfor_pickM_tL_putM(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'turnLeft'):
            if((gen_code[index-1] == 'pickMarker') and (gen_code[index+1] == 'putMarker')):
                return True
        index+=1
    return False

def checkfor_pickM_tR_putM(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'turnRight'):
            if((gen_code[index-1] == 'pickMarker') and (gen_code[index+1] == 'putMarker')):
                return True
        index+=1
    return False

def checkfor_putM_tL_pickM(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'turnLeft'):
            if((gen_code[index-1] == 'putMarker') and (gen_code[index+1] == 'pickMarker')):
                return True
        index+=1
    return False

def checkfor_putM_tR_pickM(gen_code):
    index = 1
  #  print("gen_code", gen_code)
    for token in gen_code[1:-1]:
        if(token == 'turnRight'):
            if((gen_code[index-1] == 'putMarker') and (gen_code[index+1] == 'pickMarker')):
                return True
        index+=1
    return False

def checkfor_putM_pickM(gen_code):
    index = 1
    for token in gen_code[1:-1]:
        if(token == 'putMarker'):
            if((gen_code[index-1] == 'pickMarker') or (gen_code[index+1] == 'pickMarker')):
                return True
        index+=1
    return False

def checkNextCtrl(subprog, index):
    ifelse_started = False
    for token in subprog:
        if(token == 'REPEAT'):
            return 0,index
        elif(token == 'WHILE'):
            return 0,index
        elif(token == 'IF'):
            return 0,index
        elif(token == 'IFELSE'):
            ifelse_started = True
        elif(token == 'ELSE'):
            return 0,index
        elif ((token in control_close) and (ifelse_started ==False)):
            return 1,index
        index+=1
        
def checkForOuterIf(prog):
    index = 0
    outer_if = False
    ctrl_index = 0
    
    while(index < len(prog)):
        token = prog[index]
        
        if((token == 'REPEAT') or (token == 'WHILE') or (token == 'IF') or (token == 'IFELSE') or (token == 'ELSE')):
            if(ctrl_index > 2):
                return outer_if
            
            
            if(token == 'IF'):
                outer_if = True
            value, index = checkNextCtrl(prog[index+1:], index+1)
            if(ctrl_index == 0):
                if(value != 1):
                    value, index = checkNextCtrl(prog[index+1:], index+1)
                    value, index = checkNextCtrl(prog[index+1:], index+1)
            if(not(token == 'IFELSE')):
                ctrl_index+=1
        index+=1
    return outer_if
    
def checkQuality(simulator, prg_ast_json, max_iterations):
  
        code_json = iclr18_codejson_to_karelgym_codejson(prg_ast_json)

        code = Code('karel', code_json)
        scores = []

        score = obtain_karel_saturation_score_for_code(code, max_iterations)
        scores.append(score)

        avg_score = np.mean(scores)
        return avg_score
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
    description='Random code generation from given input specification')    
    add_args(parser)
    args = parser.parse_args()

    result_dir = Path(args.data_dir)
    if not result_dir.exists():
        os.makedirs(str(result_dir))
    
    log_path = os.path.join(args.data_dir, "{}.txt".format('train_log'))
    log = ""
    
    text_path = os.path.join(args.data_dir, "{}.txt".format('train'))
    text = ""

    token_beg = ['DEF', 'run', 'm(']
    token_end = ['m)' ] 
    max_repeat =  [i for i in range(2,11)]
    min_no_actions = 2
    max_no_actions = 15
    
    final_codes = []
    simulator,tgt_tkn2idx = return_sim(args.vocab)
    numb_feat_vectors = 0

    unique_count_5 = 0
    unique_count_10 = 0
    unique_count_50 = 0
    unique_count_90 = 0
    # clear the data in the info file
    with open(log_path,'w') as file:
        pass
    
    with open(text_path,'w') as file:
        pass
        
    ctypes_to_parse = required_ctypes[args.min_index: args.max_index]
    nb_action_index = args.min_index
    for code_type in ctypes_to_parse: 
        print("code_type", required_ctypes.index(code_type))
        log = "code_type " + str(required_ctypes.index(code_type))  + "\n"
        with open(log_path, 'a+') as f:
                f.write(log)
        ctrl_count = 0
        ctrl_all_count = 0
        for token in code_type:
            if('ctrl' in token):
                ctrl_count += 1
                ctrl_all_count += 1
            if(('cif' in token) or ('celse' in token)):
                ctrl_all_count +=1
        print("nb_action_index", nb_action_index)
        
        for nb_actions in n_action_list[nb_action_index]:
            
        #all_perm = ([p for p in product(commands, repeat=ctrl_count)])
        # for selected_ctrl in all_perm:
            numb_for_code_type = 0
            #if(commands[2] in selected_ctrl): ##only if ctrl types
                
            current_spec_codes = []
            selected_ctrl = []
            for i in range(0, (args.num_codes_per_spec)):
                numb_feat_vectors +=1
                parse_success = False
                outer_if = False
                quality_good = False
                qual_bad = True
                nb_attempts = 0
                best_score = -1.0

                ##For generation
                log = ""
                log += "Numb_Actions: " + str(nb_actions)  + "\n"
                text = ""
                start = time.time()
                if(args.data_generator):
                    while(not quality_good):
                        random_code = generateCodes(code_type, list(selected_ctrl), nb_actions)
                        outer_if= checkForOuterIf(random_code)
                        if(outer_if):
                            break
                        qual_bad_1 = checkfor_tL_tR(random_code)
                        qual_bad_2 = checkfor_putM_pickM(random_code)
                        qual_bad = qual_bad_1 or qual_bad_2
                        if(not qual_bad):
                            random_code_idces = translate(random_code, tgt_tkn2idx)
                            parse_success, _,random_code_ast_json = simulator.get_prog_ast(random_code_idces)
                            if(parse_success and (not(random_code in current_spec_codes))):
                                current_spec_codes.append(random_code)   
                                quality_score = checkQuality(simulator, random_code_ast_json, args.max_iterations)
                                nb_attempts +=1
                            #    log += "Numb_Attempts: " + str(nb_attempts)  + "\n"
                            #    log += "Code " + str(random_code)  + "\n"
                            #    log += "quality_score " + str(quality_score)  + "\n"
                            #    print("quality_score",quality_score)
                                if(quality_score > args.quality_threshold):
                                    quality_good = True
                                else:
                                    
                                    if(quality_score> best_score):
                                        best_score = quality_score
                                        best_code = random_code
                                        
                                if(nb_attempts == 50):
                                    quality_good = True
                                #   random_code = best_code
                else:
                    while(not parse_success):
                        random_code = generateCodes(code_type, list(selected_ctrl), nb_actions)
                        print("random_code", random_code)
                        outer_if= checkForOuterIf(random_code)
                        
                        if(outer_if):
                            break
                        
                        #qual_bad_ = checkfor_tL_tR(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_tL_tL_tL(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_tR_tR_tR(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_tL_pickM_tR(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_tR_pickM_tL(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_tL_putM_tR(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_tR_putM_tL(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_pickM_tL_putM(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_pickM_tR_putM(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_putM_tL_pickM(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_putM_tR_pickM(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_putM_pickM(random_code)
                        
                        #qual_bad_ = qual_bad_ or checkfor_pre_REPEAT_1(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_pre_REPEAT_2(random_code)
                     ##   qual_bad_ = qual_bad_ or checkfor_pre_REPEAT_3(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_post_REPEAT_1(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_post_REPEAT_2(random_code)
                     ##   qual_bad_ = qual_bad_ or checkfor_post_REPEAT_3(random_code)
                        
                        #qual_bad_ = qual_bad_ or checkfor_notfrontIsClear_move(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_notrightIsClear_tR(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_notleftIsClear_tL(random_code)
                        
                        #qual_bad_ = qual_bad_ or checkfor_frontIsClear_tL_tR(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_rightIsClear_tL(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_leftIsClear_tR(random_code)
                        #qual_bad_ = qual_bad_ or checkfor_nomarkerPresent_pickMarker(random_code)
                        
                        #if(not qual_bad_):
                        random_code_idces = translate(random_code, tgt_tkn2idx)
                        parse_success, _, _ = simulator.get_prog_ast(random_code_idces)
                        
                end = time.time()
                
                if(not(nb_attempts == 50) and (not outer_if)):
                    log += "Numb_Attempts: " + str(nb_attempts)  + "\n"
                
                    final_codes.append(random_code)
                    log += "Code " + str(random_code)  + "\n"
                    with open(log_path, 'a+') as f:
                        f.write(log)
                        
                    text += str(random_code)  + "\n"
                    with open(text_path, 'a+') as f:
                        f.write(text)
                
                ##For evaluation
                #random_code = generateCodes(code_type, list(selected_ctrl), nb_actions)
                #parse_success, _ = simulator.get_prog_ast(random_code)
                #if(parse_success):
                    #final_codes.append(random_code)
                    #text += str(random_code)  + "\n"
                    #numb_for_code_type +=1
                        
                if(numb_for_code_type>4):
                    unique_count_5+=1
                if(numb_for_code_type>9):
                    unique_count_10+=1
                if(numb_for_code_type>49):
                    unique_count_50+=1
                if(numb_for_code_type>89):
                    unique_count_90+=1
        
        nb_action_index+=1
            
    numb_unique = len(set(map(tuple, final_codes)))   
    numb_feat = len(final_codes)
    total_generated = numb_feat_vectors

    print("numb_unique", numb_unique)
    print("numb_feat", numb_feat)
    print("total_generated", total_generated)
    print("percentage unique", (100*numb_unique/ total_generated))
    print("percentage feature", (100*numb_feat/ total_generated))

    print("unique_count_5", unique_count_5)
    print("unique_count_10", unique_count_10)
    print("unique_count_50", unique_count_50)
    print("unique_count_90", unique_count_90)
               
                
    
    
    
    
