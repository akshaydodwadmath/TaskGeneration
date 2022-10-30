import numpy as np
import math
import argparse
import json
import random
actions = [
    'move',
    'turnLeft',
    'turnRight',
    'pickMarker',
    'putMarker',
]

commands = ['REPEAT',
            'WHILE',
            'IF',
            ]
command_if_else = [
            'IFELSE',
            'ELSE',
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
control_open = [
         'r(',
         'w(',
         'i(',
         'e(',
    ]


control_close = [
         'r)',
         'w)',
         'i)',
         'e)',
    ]

#FeatureVector format: [C1_D1_WHILE, C1_D1_REPEAT, C1_D1_IF, [C1_D1_IFELSE, C1_D1_ELSE] ,C1_D2_WHILE, C1_D2_REPEAT, C1_D2_IF, [C1_D2_IFELSE, C1_D2_ELSE] , 
#                       C2_D1_WHILE ,C2_D1_REPEAT, C2_D1_IF, [C2_D1_IFELSE, C2_D1_ELSE] ,C2_D2_WHILE ,C2_D2_REPEAT, C2_D2_IF, [C2_D2_IFELSE, C2_D2_ELSE]]
required_ctypes = [['action1'],
                   
                   ['action1', 'commands', 'cond', 'control_open', 'action2', 'control_close', 'action3'] ] 
                   #'(D_IF()D_ELSE())',
                   
                   #'(D_CTRL()D_CTRL())',
                   #'(D_IF()D_ELSE()D_CTRL())',
                   #'(D_CTRL()D_IF()D_ELSE())',
                   #'(D_IF()D_ELSE()D_IF()D_ELSE())',
                   
                   #'(D_CTRL(D_CTRL()))',
                   #'(D_CTRL(D_IF()D_ELSE()))',
                   #'(D_IF(D_CTRL())D_ELSE())',
                   #'(D_IF(D_IF()D_ELSE())D_ELSE())',
                   #'(D_IF()D_ELSE(D_CTRL()))',
                   #'(D_IF()D_ELSE(D_IF()D_ELSE()))',

                   #'(D_IF(D_CTRL())D_ELSE(D_CTRL()))',
                   #'(D_IF(D_IF()D_ELSE())D_ELSE(D_CTRL()))',
                   #'(D_IF(D_CTRL())D_ELSE(D_IF()D_ELSE()))',
                   #'(D_IF(D_IF()D_ELSE())D_ELSE(D_IF()D_ELSE()))']
                   #"CodeType": "FeatureVector"
                   
                   #'( D_CTRL ( ) D_CTRL ( ) D_CTRL ( ) )',
                   #'( D_CTRL ( ) D_CTRL ( D_CTRL ( ) ) )',
                   #'( D_CTRL ( ) D_CTRL ( D_CTRL ( D_CTRL ( ) ) ) )',
                   #'( D_CTRL ( ) D_CTRL ( D_IF ( ) D_ELSE ( ) ) )',
                   #'( D_CTRL ( ) D_IF ( ) D_ELSE ( ) )',
                   #'( D_CTRL ( ) D_IF ( D_CTRL ( ) ) D_ELSE ( ) )',
                   #'( D_CTRL ( D_CTRL ( ) ) )',
                   #'( D_CTRL ( D_CTRL ( ) ) D_CTRL ( ) )',
                   #'( D_CTRL ( D_CTRL ( ) ) D_CTRL ( ) D_CTRL ( ) )'
                   
token_beg = ['DEF', 'run', 'm(']
token_end = [')m' ] 

max_actions = [i for i in range(2,17)]
max_repeat =  [i for i in range(2,11)]

def generate_codes(code_type, max_nb_actions):
    code= []

    code += token_beg

    action_set = random.choices(actions, k=max_nb_actions)
    print(action_set)
    action_in_code_type = []
    
    for token in code_type:
        if (('action' in token)):
            code.append(token)
           # action_in_code_type.append(token)
            #action_count +=1
        elif(token == 'commands'):
            code.append(random.choice(commands))
        elif(token == 'cond'):
            rand_cond = random.choice(cond)
            if(rand_cond == cond[-1]):# for repeat
                rand_cond = [str(rand_cond[0]) + str(random.choice(max_repeat))]
            code += rand_cond
        elif(token == 'control_open'):
            code.append(random.choice(control_open))
        elif(token == 'control_close'):
            code.append(random.choice(control_close))
  #  print(code)
    
    
    while(action_set):
        index = 0
        for token in code:
            if(np.random.choice(2, 1) == 1): #prob = 0.25
                if('action' in token):
                    code[index] = action_set.pop()
                elif(token in actions):
                    code.insert(index+1, action_set.pop())
                    
                if(not action_set):
                    break
            index+=1
            
    for token in code:
        if ('action' in token):
            code.remove(token)
 #   print("code_updated", code)
 #   print("action_set", action_set)

    code += token_end
    code = list(filter(None, code))
    return code

for code_type in required_ctypes: 
    for nb_actions in max_actions:
        random_code = generate_codes(code_type, nb_actions)
        print(random_code)
        print(nb_actions)

               
                
    
    
    
    
