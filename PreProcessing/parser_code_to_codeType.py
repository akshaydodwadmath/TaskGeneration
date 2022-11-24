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
required_ctypes = [
                    '()', #Type01 #CT1
                   
                   '(D_CTRL())', #Type02 #CT2
                   '(D_IF()D_ELSE())', #Type03 #CT2
                   
                   '(D_CTRL()D_CTRL())', #Type04 #CT3
                   '(D_IF()D_ELSE()D_CTRL())', #Type05  #CT3
                   '(D_CTRL()D_IF()D_ELSE())', #Type06 #CT3
                   '(D_IF()D_ELSE()D_IF()D_ELSE())', #Type07 #CT3
                   
                   '(D_CTRL(D_CTRL()))', #Type08 #CT4
                   '(D_CTRL(D_IF()D_ELSE()))', #Type09 #CT4
                   '(D_IF(D_CTRL())D_ELSE())', #Type10 #CT4 
                   '(D_IF(D_IF()D_ELSE())D_ELSE())', #Type11 #CT4
                   '(D_IF()D_ELSE(D_CTRL()))', #Type12 #CT4
                   '(D_IF()D_ELSE(D_IF()D_ELSE()))', #Type13 #CT4
                   
                   '(D_IF(D_CTRL())D_ELSE(D_CTRL()))', #Type14 #CT4
                   '(D_IF(D_IF()D_ELSE())D_ELSE(D_CTRL()))', #Type15 #CT4
                   '(D_IF(D_CTRL())D_ELSE(D_IF()D_ELSE()))', #Type16 #CT4
                   '(D_IF(D_IF()D_ELSE())D_ELSE(D_IF()D_ELSE()))',#Type17 #CT4
                   
                   '(D_CTRL(D_CTRL())D_CTRL())',#Type18 #CT5
                   '(D_CTRL(D_IF()D_ELSE())D_CTRL())',#Type19 #CT5
                   '(D_CTRL(D_CTRL())D_IF()D_ELSE())',#Type20 #CT5
                   '(D_CTRL(D_IF()D_ELSE())D_IF()D_ELSE())',#Type21 #CT5
                   
                   '(D_IF(D_CTRL())D_ELSE()D_CTRL())',#Type22 #CT5
                   '(D_IF(D_IF()D_ELSE())D_ELSE()D_CTRL())',#Type23 #CT5
                   '(D_IF(D_CTRL())D_ELSE()D_IF()D_ELSE())',#Type24 #CT5
                   '(D_IF(D_IF()D_ELSE())D_ELSE()D_IF()D_ELSE())',#Type25 #CT5
                   
                   '(D_IF()D_ELSE(D_CTRL())D_CTRL())',#Type26 #CT5
                   '(D_IF()D_ELSE(D_IF()D_ELSE())D_CTRL())',#Type27 #CT5
                   '(D_IF()D_ELSE(D_CTRL())D_IF()D_ELSE())',#Type28 #CT5
                   '(D_IF()D_ELSE(D_IF()D_ELSE())D_IF()D_ELSE())',#Type29 #CT5
                   
                   '(D_CTRL()D_CTRL(D_CTRL()))',#Type30 #CT6
                   '(D_CTRL()D_CTRL(D_IF()D_ELSE()))',#Type31 #CT6
                   '(D_IF()D_ELSE()D_CTRL(D_CTRL()))',#Type32 #CT6
                   '(D_IF()D_ELSE()D_CTRL(D_IF()D_ELSE()))',#Type33 #CT6
                   
                   '(D_CTRL()D_IF(D_CTRL())D_ELSE())',#Type34 #CT6
                   '(D_CTRL()D_IF(D_IF()D_ELSE())D_ELSE())',#Type35 #CT6
                   '(D_IF()D_ELSE()D_IF(D_CTRL())D_ELSE())',#Type36 #CT6
                   '(D_IF()D_ELSE()D_IF(D_IF()D_ELSE())D_ELSE())',#Type37 #CT6
                   
                   '(D_CTRL()D_IF()D_ELSE(D_CTRL()))',#Type38 #CT6
                   '(D_CTRL()D_IF()D_ELSE(D_IF()D_ELSE()))',#Type39 #CT6
                   '(D_IF()D_ELSE()D_IF()D_ELSE(D_CTRL()))',#Type40 #CT6
                   '(D_IF()D_ELSE()D_IF()D_ELSE(D_IF()D_ELSE()))',#Type41 #CT6
                   
                   '(D_CTRL()D_CTRL()D_CTRL())',#Type42 #CT7
                   '(D_CTRL()D_CTRL()D_IF()D_ELSE())',#Type43 #CT7
                   '(D_CTRL()D_IF()D_ELSE()D_CTRL())',#Type44 #CT7                  
                   '(D_IF()D_ELSE()D_CTRL()D_CTRL())',#Type45 #CT7
                   '(D_CTRL()D_IF()D_ELSE()D_IF()D_ELSE())',#Type46 #CT7
                   '(D_IF()D_ELSE()D_IF()D_ELSE()D_CTRL())',#Type47 #CT7
                   '(D_IF()D_ELSE()D_CTRL()D_IF()D_ELSE())',#Type48 #CT7
                   '(D_IF()D_ELSE()D_IF()D_ELSE()D_IF()D_ELSE())',#Type49 #CT7
                   
                   '(D_CTRL(D_CTRL()D_CTRL()))', #Type50 #CT8
                   '(D_CTRL(D_IF()D_ELSE()D_CTRL()))', #Type51  #CT8
                   '(D_CTRL(D_CTRL()D_IF()D_ELSE()))', #Type52 #CT8
                   '(D_CTRL(D_IF()D_ELSE()D_IF()D_ELSE()))', #Type53 #CT8
                   
                   '(D_IF(D_CTRL()D_CTRL())D_ELSE())', #Type54 #CT8
                   '(D_IF(D_IF()D_ELSE()D_CTRL())D_ELSE())', #Type55  #CT8
                   '(D_IF(D_CTRL()D_IF()D_ELSE())D_ELSE())', #Type56 #CT8
                   '(D_IF(D_IF()D_ELSE()D_IF()D_ELSE())D_ELSE())', #Type57 #CT8
                   
                   '(D_IF()D_ELSE(D_CTRL()D_CTRL()))', #Type58 #CT8
                   '(D_IF()D_ELSE(D_IF()D_ELSE()D_CTRL()))', #Type59  #CT8
                   '(D_IF()D_ELSE(D_CTRL()D_IF()D_ELSE()))', #Type60 #CT8
                   '(D_IF()D_ELSE(D_IF()D_ELSE()D_IF()D_ELSE()))', #Type61 #CT8
                   ]




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
    subVec_format = ['NO_CTRL', 'REPEAT','WHILE','IF','IFELSE']
    featVec = [[0,[[0,0,0,0,0],[0,0,0,0,0]],0,[[0,0,0,0,0],[0,0,0,0,0]],0,[[0,0,0,0,0],[0,0,0,0,0]],0,[[0,0,0,0,0],[0,0,0,0,0]],[[0,0,0,0,0],[0,0,0,0,0]]],
               [0,[0,0,0,0,0],0,[0,0,0,0,0],0,[0,0,0,0,0],0,[0,0,0,0,0],[0,0,0,0,0]],
               [0,[0,0,0,0,0],0,[0,0,0,0,0],0,[0,0,0,0,0],0,[0,0,0,0,0],[0,0,0,0,0]],
               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    #for token in prog[index:]:
    ctrl_index = 0
    numb_actions = getNumberOfActions(prog)
    while(index < len(prog)):
        token = prog[index]
        if(token == 'REPEAT'):
            featVec[ctrl_index][featVec_format.index('REPEAT')] = 1
            value, index = checkNextCtrl(prog[index+1:], index+1)
            if(ctrl_index == 0):
                featVec[ctrl_index][featVec_format.index('REPEAT')+1][0][value] = 1
                if(value != subVec_format.index('NO_CTRL')):
                    value, index = checkNextCtrl(prog[index+1:], index+1)
                    value, index = checkNextCtrl(prog[index+1:], index+1)
                    featVec[ctrl_index][featVec_format.index('REPEAT')+1][1][value] = 1
                else:
                    featVec[ctrl_index][featVec_format.index('REPEAT')+1][1][subVec_format.index('NO_CTRL')] = 1
            else:
                featVec[ctrl_index][featVec_format.index('REPEAT')+1][value] = 1
            ctrl_index+=1
        elif(token == 'WHILE'):
            featVec[ctrl_index][featVec_format.index('WHILE')] = 1
            value, index = checkNextCtrl(prog[index+1:], index+1)
            if(ctrl_index == 0):
                featVec[ctrl_index][featVec_format.index('WHILE')+1][0][value] = 1
                if(value != subVec_format.index('NO_CTRL')):
                    value, index = checkNextCtrl(prog[index+1:], index+1)
                    value, index = checkNextCtrl(prog[index+1:], index+1)
                    featVec[ctrl_index][featVec_format.index('WHILE')+1][1][value] = 1
                else:
                    featVec[ctrl_index][featVec_format.index('WHILE')+1][1][subVec_format.index('NO_CTRL')] = 1
            else:
                featVec[ctrl_index][featVec_format.index('WHILE')+1][value] = 1
            ctrl_index+=1
        elif(token == 'IF'):
            featVec[ctrl_index][featVec_format.index('IF')] = 1
            value, index = checkNextCtrl(prog[index+1:], index+1)
            if(ctrl_index == 0):
                featVec[ctrl_index][featVec_format.index('IF')+1][0][value] = 1
                if(value != subVec_format.index('NO_CTRL')):
                    value, index = checkNextCtrl(prog[index+1:], index+1)
                    value, index = checkNextCtrl(prog[index+1:], index+1)
                    featVec[ctrl_index][featVec_format.index('IF')+1][1][value] = 1
                else:
                    featVec[ctrl_index][featVec_format.index('IF')+1][1][subVec_format.index('NO_CTRL')] = 1
            else:
                featVec[ctrl_index][featVec_format.index('IF')+1][value] = 1
            ctrl_index+=1
        elif(token == 'IFELSE'):
            featVec[ctrl_index][featVec_format.index('IFELSE')] = 1
            value, index = checkNextCtrl(prog[index+1:], index+1)
            if(ctrl_index == 0):
                featVec[ctrl_index][featVec_format.index('IFELSE')+1][0][value] = 1
                if(value != subVec_format.index('NO_CTRL')):
                    value, index = checkNextCtrl(prog[index+1:], index+1) #For closing bracket
                    value, index = checkNextCtrl(prog[index+1:], index+1)
                    featVec[ctrl_index][featVec_format.index('IFELSE')+1][1][value] = 1
                else:
                    featVec[ctrl_index][featVec_format.index('IFELSE')+1][1][subVec_format.index('NO_CTRL')] = 1
            else:
                featVec[ctrl_index][featVec_format.index('IFELSE')+1][value] = 1
        elif(token == 'ELSE'):
            value, index = checkNextCtrl(prog[index+1:], index+1)
            if(ctrl_index == 0):
                featVec[ctrl_index][featVec_format.index('IFELSE')+2][0][value] = 1
                if(value != subVec_format.index('NO_CTRL')):
                    value, index = checkNextCtrl(prog[index+1:], index+1) #For closing bracket
                    value, index = checkNextCtrl(prog[index+1:], index+1)
                    featVec[ctrl_index][featVec_format.index('IFELSE')+2][1][value] = 1
                else:
                    featVec[ctrl_index][featVec_format.index('IFELSE')+2][1][subVec_format.index('NO_CTRL')] = 1
            else:
                featVec[ctrl_index][featVec_format.index('IFELSE')+2][value] = 1
            ctrl_index+=1
        index+=1
    featVec[3][numb_actions] = 1
    return featVec, numb_actions

def add_args(parser):
    
    parse_group = parser.add_argument_group("Conversion",
                                        description="Conversion options")
    
    
    parse_group.add_argument("--input_code_file", type=str,
                            default="code_val.txt",
                            help="Path to the input code file. "
                            " Default: %(default)s")
    
    
    parse_group.add_argument("--code_type_file", type=str,
                            default="code_type_val.txt",
                            help="Path to the input code file. "
                            " Default: %(default)s")
    parse_group.add_argument("--json_data_file", type=str,
                            default="val_data.json",
                            help="Path to the input code file. "
                            " Default: %(default)s")
    parse_group.add_argument("--json_featVectors_file", type=str,
                            default="featVectors.json",
                            help="Path to the input code file. "
                            " Default: %(default)s")
    parse_group.add_argument('--ndomains', type=int,
                            default=5)


parser = argparse.ArgumentParser(
    description='Convert code to code types.')
add_args(parser)
args = parser.parse_args()

code_file = open(args.input_code_file, 'r')

code_type_file = open(args.code_type_file, "w")

Lines = code_file.readlines() 

line_count = 0
code_type_list = []
#dict_obj = my_dictionary()
list_obj = []
featVec_evaluation = []
feat_vect_elements = []
no_of_vect_elements =  []
action_temp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

code_list = []
for line in Lines:
    
    prog_updated = []
    
    prog = line.split(" ")
    for token in prog:
    #    token = token.replace('[', '')
    #    token = token.replace(']', '')
        token = token.replace("'", "")
        token = token.replace('"', '')
        token = token.replace(' ', '')
        token = token.replace(',', '')
        token = token.replace('[', '')
        token = token.replace(']', '')
        token = token.replace('\n', '')
        prog_updated.append(token)
        #Example format: ['DEF', 'run', 'm(', 'move', 'move', 'WHILE', 'c(', 'noMarkersPresent', 'c)', 'w(', 'putMarker', 'move', 'w)', 'm)']


    #print("prog", prog_updated)
    
    pattern_only = []
    
    for token in prog_updated:
        if(token in commands):
            pattern_only.append("D_CTRL" )
        if(token == command_if_else[0]):
            pattern_only.append("D_" + "IF")
        if(token == command_if_else[1]):
            pattern_only.append("D_" + "ELSE")
        if(token in control_open):
            pattern_only.append("(")
        if(token in control_close):
            pattern_only.append(")")
            
            
   # print("pattern_only", pattern_only)
    code_type = "".join(pattern_only) 
    
    code_type_list.append(code_type)
    code_type_file.write( code_type + "\n" )

    if(code_type in required_ctypes):
        
        feat_vect,numb_actions = getFeatureVector(prog_updated)
        action_temp[numb_actions] +=1
        if(feat_vect not in feat_vect_elements):
            feat_vect_elements.append(feat_vect)
            no_of_vect_elements.append(0)
            code_list.append([0] * args.ndomains)
            
        no_of_vect_elements[feat_vect_elements.index(feat_vect)] +=1
        
        if( no_of_vect_elements[feat_vect_elements.index(feat_vect)] <= args.ndomains):
            if(prog_updated in code_list[feat_vect_elements.index(feat_vect)][:]):
                 no_of_vect_elements[feat_vect_elements.index(feat_vect)] -=1
            else:
                code_list[feat_vect_elements.index(feat_vect)][no_of_vect_elements[feat_vect_elements.index(feat_vect)] - 1] = prog_updated 
        
        if( no_of_vect_elements[feat_vect_elements.index(feat_vect)] == args.ndomains):
            featVec_evaluation.append({
                            "CodeTypeIndex": required_ctypes.index(code_type),
                            "FeatureVector":feat_vect,
                            "FeatureVectorIndex":feat_vect_elements.index(feat_vect),
                            "Code": code_list[feat_vect_elements.index(feat_vect)][0]
                            })
            
            for i in range(0,args.ndomains):
                list_obj.append({"CodeTypeIndex": required_ctypes.index(code_type),
                                "FeatureVector":feat_vect,
                                "FeatureVectorIndex":feat_vect_elements.index(feat_vect),
                                "Code": code_list[feat_vect_elements.index(feat_vect)][i]})
              
        
                line_count+=1
  #  if(line_count>1):
  #      break
    
values, counts = np.unique(code_type_list, return_counts=True)    

with open(args.json_featVectors_file, 'w') as json_file:
    for dict_ in featVec_evaluation:
        json.dump(dict_, json_file, 
                        separators=(',',': '))
        json_file.write('\n')

with open(args.json_data_file, 'w') as json_file:
    for dict_ in list_obj:
        json.dump(dict_, json_file, 
                        separators=(',',': '))
        json_file.write('\n')
    
    
print("Total count\n", line_count)
print("Number of feature vectors\n", len(no_of_vect_elements))
print("Action temp\n", action_temp)
#dict_obj = {}
#for v, c in zip(values, counts):
    
    
    #dict_obj[v] = c

    
#sorted_dict_obj = {k: v for k, v in sorted(dict_obj.items(), key=lambda item: item[1], reverse=True)[:100]}
#print("sorted_dict_obj",sorted_dict_obj) 
#print("len", len(counts))
    
    
    
    
