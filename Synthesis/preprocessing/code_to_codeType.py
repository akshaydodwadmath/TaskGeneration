import numpy as np
import math
import argparse
import json

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
command_if_else = ['IFELSE',
            'ELSE',
            ]
control_open = ['m(',
         'r(',
         'w(',
         'i(',
         'e(',
    ]


control_close = ['m)',
         'r)',
         'w)',
         'i)',
         'e)',
    ]

#FeatureVector format: [C1_D1_WHILE, C1_D1_REPEAT, C1_D1_IF, [C1_D1_IFELSE, C1_D1_ELSE] ,C1_D2_WHILE, C1_D2_REPEAT, C1_D2_IF, [C1_D2_IFELSE, C1_D2_ELSE] , 
#                       C2_D1_WHILE ,C2_D1_REPEAT, C2_D1_IF, [C2_D1_IFELSE, C2_D1_ELSE] ,C2_D2_WHILE ,C2_D2_REPEAT, C2_D2_IF, [C2_D2_IFELSE, C2_D2_ELSE]]
required_ctypes = {'()',
                   
                   '(D_CTRL())', 
                   '(D_IF()D_ELSE())',
                   
                   '(D_CTRL()D_CTRL())',
                   '(D_IF()D_ELSE()D_CTRL())',
                   '(D_CTRL()D_IF()D_ELSE())',
                   '(D_IF()D_ELSE()D_IF()D_ELSE())',
                   
                   '(D_CTRL(D_CTRL()))',
                   '(D_CTRL(D_IF()D_ELSE()))',
                   '(D_IF(D_CTRL())D_ELSE())',
                   '(D_IF(D_IF()D_ELSE())D_ELSE())',
                   '(D_IF()D_ELSE(D_CTRL()))',
                   '(D_IF()D_ELSE(D_IF()D_ELSE()))',
                   
                   '(D_IF(D_CTRL())D_ELSE(D_CTRL()))',
                   '(D_IF(D_IF()D_ELSE())D_ELSE(D_CTRL()))',
                   '(D_IF(D_CTRL())D_ELSE(D_IF()D_ELSE()))',
                   '(D_IF(D_IF()D_ELSE())D_ELSE(D_IF()D_ELSE()))'} 
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
               
                   

def checkNextCtrl(subprog, index):
    ifelse_started = False
    subVec_format = ['NO_CTRL', 'REPEAT','WHILE','IF','IFELSE']
    for token in subprog:
        if(token == 'REPEAT'):
            return(subVec_format.index('REPEAT'), index)
        elif(token == 'WHILE'):
            return(subVec_format.index('WHILE'), index)
        elif(token == 'IF'):
            return(subVec_format.index('IF'), index)
        elif(token == 'IFELSE'):
            ifelse_started = True
        elif(token == 'ELSE'):
            return(subVec_format.index('IFELSE'), index)
        elif ((token in control_close) and (ifelse_started ==False)):
            return(subVec_format.index('NO_CTRL'), index)
        index+=1

def getNumberOfActions(prog):
    action_count = 0
    for token in prog:
        if(token in actions):
            action_count+=1
    return action_count
    

def getFeatureVector(prog):
    index = 0
    ########[No_Const, Repeat, [],While,[],If,[],IFELSE,[],[]]
    featVec_format = ['REPEAT', ['NO_CTRL', 'REPEAT','WHILE','IF','IFELSE'],'WHILE',['NO_CTRL', 'REPEAT','WHILE','IF','IFELSE'],
                      'IF',  ['NO_CTRL', 'REPEAT','WHILE','IF','IFELSE'],'IFELSE', ['NO_CTRL', 'REPEAT','WHILE','IF','IFELSE'],['NO_CTRL', 'REPEAT','WHILE','IF','IFELSE']]
    featVec = [[0,[0,0,0,0,0],0,[0,0,0,0,0],0,[0,0,0,0,0],0,[0,0,0,0,0],[0,0,0,0,0]],
               [0,[0,0,0,0,0],0,[0,0,0,0,0],0,[0,0,0,0,0],0,[0,0,0,0,0],[0,0,0,0,0]],
               [0,[0,0,0,0,0],0,[0,0,0,0,0],0,[0,0,0,0,0],0,[0,0,0,0,0],[0,0,0,0,0]],
               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    featVec_undefined = [[1,[1,1,1,1,1],1,[1,1,1,1,1],1,[1,1,1,1,1],1,[1,1,1,1,1],[1,1,1,1,1]],
               [1,[1,1,1,1,1],1,[1,1,1,1,1],1,[1,1,1,1,1],1,[1,1,1,1,1],[1,1,1,1,1]],
               [1,[1,1,1,1,1],1,[1,1,1,1,1],1,[1,1,1,1,1],1,[1,1,1,1,1],[1,1,1,1,1]],
               [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
    #for token in prog[index:]:
    ctrl_index = 0
    numb_actions = getNumberOfActions(prog)
    while(index < len(prog)):
        token = prog[index]
        if((token in commands) or (token in command_if_else)):
            if(ctrl_index > 2):
                return featVec_undefined, numb_actions
        
        if(token == 'REPEAT'):
            featVec[ctrl_index][featVec_format.index('REPEAT')] = 1
            value, index = checkNextCtrl(prog[index+1:], index+1)
            featVec[ctrl_index][featVec_format.index('REPEAT')+1][value] = 1
            ctrl_index+=1
        elif(token == 'WHILE'):
            featVec[ctrl_index][featVec_format.index('WHILE')] = 1
            value, index = checkNextCtrl(prog[index+1:], index+1)
            featVec[ctrl_index][featVec_format.index('WHILE')+1][value] = 1
            ctrl_index+=1
        elif(token == 'IF'):
            featVec[ctrl_index][featVec_format.index('IF')] = 1
            value, index = checkNextCtrl(prog[index+1:], index+1)
            featVec[ctrl_index][featVec_format.index('IF')+1][value] = 1
            ctrl_index+=1
        elif(token == 'IFELSE'):
            featVec[ctrl_index][featVec_format.index('IFELSE')] = 1
            value, index = checkNextCtrl(prog[index+1:], index+1)
            featVec[ctrl_index][featVec_format.index('IFELSE')+1][value] = 1
        elif(token == 'ELSE'):
            value, index = checkNextCtrl(prog[index+1:], index+1)
            featVec[ctrl_index][featVec_format.index('IFELSE')+2][value] = 1
            ctrl_index+=1
        index+=1
    featVec[3][numb_actions] = 1
    return featVec, numb_actions
