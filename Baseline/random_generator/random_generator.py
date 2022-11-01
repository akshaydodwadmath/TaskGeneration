import numpy as np
import math
import argparse
import json
import random
import os

from karel.consistency import Simulator


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
    ['c(','not','c(','markersPresent','c)','c)'],
    ['c(','not','c(','noMarkersPresent','c)','c)'],
    ['c(','not','c(','leftIsClear','c)','c)'],
    ['c(','not','c(','rightIsClear','c)','c)'],
    ['c(','not','c(','frontIsClear','c)','c)'],
    ['R=']# repeat options 2 to 10
    ]

required_ctypes = [['action'], #Type01
                   
                   ['action', 
                    'ctrl1', 'cond1', 'copen1', 'action', 'cclose1', 
                    'action' ], #Type02
                   
                   ['action', 
                    'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                    'celse', 'c_elseopen', 'action', 'c_elseclose', 
                    'action'], #Type03
                   ['action', 
                     'ctrl1', 'cond1', 'copen1', 'action', 'cclose1',
                     'action', 
                     'ctrl2', 'cond2', 'copen2', 'action', 'cclose2', 
                     'action'],#Type04
                    ['action', 
                     'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                    'celse', 'c_elseopen', 'action', 'c_elseclose', 
                    'action'],#Type05
                    ['action', 
                     'ctrl1', 'cond1', 'copen1', 'action', 'cclose1',
                     'action', 
                    'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                    'celse', 'c_elseopen', 'action', 'c_elseclose', 
                    'action'],#Type06
                    ['action', 
                    'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                    'celse', 'c_elseopen', 'action', 'c_elseclose',  
                    'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                    'celse', 'c_elseopen', 'action', 'c_elseclose', 
                    'action'],#Type07
                   
                    ['action', 
                     'ctrl1', 'cond1', 'copen1', 
                     'ctrl2', 'cond2', 'copen2', 'action', 'cclose2', 
                     'action', 'cclose1', 
                     'action'], #Type 08
                    ['action', 
                     'ctrl1', 'cond1', 'copen1', 
                     'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                     'celse', 'c_elseopen', 'action', 'c_elseclose', 
                     'action', 'cclose1',
                     'action'],
                    ['action', #Type 09
                     'cif', 'c_cndif', 'c_ifopen', 
                     'ctrl1', 'cond1', 'copen1', 'action', 'cclose1', 
                     'action','c_ifclose',
                     'celse', 'c_elseopen', 
                     'action', 'c_elseclose', 
                     'action'], #Type 10
                    ['action', 
                     'cif', 'c_cndif', 'c_ifopen', 
                     'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                     'celse', 'c_elseopen', 'action', 'c_elseclose', 
                     'action','c_ifclose',
                     'celse', 'c_elseopen', 
                     'action', 'c_elseclose', 
                     'action'], #Type 11
                    ['action', 
                     'cif', 'c_cndif', 'c_ifopen', 
                     'action','c_ifclose',
                     'celse', 'c_elseopen', 
                     'ctrl1', 'cond1', 'copen1', 'action', 'cclose1', 
                     'action', 'c_elseclose', 
                     'action'], #Type 12
                    ['action', 
                     'cif', 'c_cndif', 'c_ifopen', 
                     'action','c_ifclose',
                     'celse', 'c_elseopen', 
                     'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                     'celse', 'c_elseopen', 'action', 'c_elseclose', 
                     'action', 'c_elseclose', 
                     'action'], #Type 13
                    
                    
                    ['action', 
                     'cif', 'c_cndif', 'c_ifopen', 
                     'ctrl1', 'cond1', 'copen1', 'action', 'cclose1', 
                     'action','c_ifclose',
                     'celse', 'c_elseopen', 
                     'ctrl2', 'cond2', 'copen2', 'action', 'cclose2', 
                     'action', 'c_elseclose', 
                     'action'], #Type 14
                    ['action', 
                     'cif', 'c_cndif', 'c_ifopen', 
                     'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                     'celse', 'c_elseopen', 'action', 'c_elseclose', 
                     'action','c_ifclose',
                     'celse', 'c_elseopen', 
                     'ctrl1', 'cond1', 'copen1', 'action', 'cclose1', 
                     'action', 'c_elseclose', 
                     'action'], #Type 15
                   
                    ['action', 
                     'cif', 'c_cndif', 'c_ifopen', 
                     'ctrl1', 'cond1', 'copen1', 'action', 'cclose1', 
                     'action','c_ifclose',
                     'celse', 'c_elseopen', 
                     'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                     'celse', 'c_elseopen', 'action', 'c_elseclose', 
                     'action', 'c_elseclose', 
                     'action'], #Type 16
                    ['action', 
                     'cif', 'c_cndif', 'c_ifopen', 
                     'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                     'celse', 'c_elseopen', 'action', 'c_elseclose', 
                     'action','c_ifclose',
                     'celse', 'c_elseopen', 
                     'cif', 'c_cndif', 'c_ifopen', 'action', 'c_ifclose',
                     'celse', 'c_elseopen', 'action', 'c_elseclose', 
                     'action', 'c_elseclose', 
                     'action']] #Type 17
    
def add_args(parser):
    parse_group = parser.add_argument_group("Conversion",
                                        description="Conversion options")
    parse_group.add_argument('--data_dir', type=str, default='data')
    
    

parser = argparse.ArgumentParser(
    description='Convert code to code types.')    
add_args(parser)
args = parser.parse_args()

text_path = os.path.join(args.data_dir, "{}.txt".format('train'))
text = ""

token_beg = ['DEF', 'run', 'm(']
token_end = ['m)' ] 
max_actions = [i for i in range(2,17)]
max_repeat =  [i for i in range(2,11)]

def generate_codes(code_type, max_nb_actions):
    
    code= []
    code += token_beg
    action_set = random.choices(actions, k=max_nb_actions)
    action_in_code_type = []
    ctrl_count = 0
    cifelse_count = 0
    for token in code_type:
        if('ctrl' in token):
            ctrl_count += 1
        code.append(token)
    ctrl_set = random.choices(commands, k=ctrl_count)

    #Add actions
    while(action_set):
        index = 0
        for token in code:
            if(np.random.choice(2, 1) == 1): #prob = 0.5
                if('action' in token):
                    code[index] = action_set.pop()
                elif(token in actions):
                    code.insert(index+1, action_set.pop())
                    
                if(not action_set):
                    break
            index+=1
    
    #Add control statements and brackets
    open_set = []
    close_set = []
    cond_set = []
    index = 0
    for token in code:
        if('ctrl' in token):
            current_ctrl_set = ctrl_set.pop()
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


#Main code
n_domains = 24
top_k = 10
final_codes = []
simulator = Simulator()

for i in range(0, (n_domains*top_k)):
    for code_type in required_ctypes: 
        for nb_actions in max_actions:
            random_code = generate_codes(code_type, nb_actions)
            parse_success, _ = simulator.get_prog_ast(random_code)
            if(parse_success):
                final_codes.append(random_code)
                text += str(random_code)  + "\n"
numb_unique = len(set(map(tuple, final_codes)))   
numb_feat = len(final_codes)
total_generated = n_domains * top_k * len(max_actions)*(len(required_ctypes))
with open(text_path, 'w') as f:
    f.write(text)
print("numb_unique", numb_unique)
print("numb_feat", numb_feat)
print("total_generated", total_generated)
print("percentage unique", (100*numb_unique/ total_generated))
print("percentage feature", (100*numb_feat/ total_generated))

               
                
    
    
    
    
