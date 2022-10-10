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
required_ctypes = {'()': '[0,[0,0,0],0,[0,0,0],0,[0,0,0]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   
                   '(D_CTRL())': '[1,[0,0,0],0,[0,0,0],0,[0,0,0]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]', 
                   '(D_IF()D_ELSE())': '[0,[0,0,0],1,[0,0,0],1,[0,0,0]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   
                   '(D_CTRL()D_CTRL())': '[1,[0,0,0],0,[0,0,0],0,[0,0,0]], [1,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   '(D_IF()D_ELSE()D_CTRL())': '[0,[0,0,0],1,[0,0,0],1,[0,0,0]], [1,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   '(D_CTRL()D_IF()D_ELSE())': '[1,[0,0,0],0,[0,0,0],0,[0,0,0]], [0,[0,0,0],1,[0,0,0],1,[0,0,0]]',
                   '(D_IF()D_ELSE()D_IF()D_ELSE())': '[0,[0,0,0],1,[0,0,0],1,[0,0,0]], [0,[0,0,0],1,[0,0,0],1,[0,0,0]]',
                   
                   '(D_CTRL(D_CTRL()))': '[1,[1,0,0],0,[0,0,0],0,[0,0,0]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   '(D_CTRL(D_IF()D_ELSE()))': '[1,[0,1,1],0,[0,0,0],0,[0,0,0]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   '(D_IF(D_CTRL())D_ELSE())': '[0,[0,0,0],1,[1,0,0],1,[0,0,0]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   '(D_IF(D_IF()D_ELSE())D_ELSE())': '[0,[0,0,0],1,[0,1,1],1,[0,0,0]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   '(D_IF()D_ELSE(D_CTRL()))': '[0,[0,0,0],1,[0,0,0],1,[1,0,0]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   '(D_IF()D_ELSE(D_IF()D_ELSE()))': '[0,[0,0,0],1,[0,0,0],1,[0,1,1]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   
                   '(D_IF(D_CTRL())D_ELSE(D_CTRL()))': '[0,[0,0,0],1,[1,0,0],1,[1,0,0]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   '(D_IF(D_IF()D_ELSE())D_ELSE(D_CTRL()))': '[0,[0,0,0],1,[0,1,1],1,[1,0,0]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   '(D_IF(D_CTRL())D_ELSE(D_IF()D_ELSE()))': '[0,[0,0,0],1,[1,0,0],1,[0,1,1]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]',
                   '(D_IF(D_IF()D_ELSE())D_ELSE(D_IF()D_ELSE()))': '[0,[0,0,0],1,[0,1,1],1,[0,1,1]], [0,[0,0,0],0,[0,0,0],0,[0,0,0]]'} 
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
                   
                   

def add_args(parser):
    
    parse_group = parser.add_argument_group("Conversion",
                                        description="Conversion options")
    
    
    parse_group.add_argument("--code_file", type=str,
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


parser = argparse.ArgumentParser(
    description='Convert code to code types.')
add_args(parser)
args = parser.parse_args()

code_file = open(args.code_file, 'r')

code_type_file = open(args.code_type_file, "w")

Lines = code_file.readlines() 

line_count = 0
code_type_list = []
#dict_obj = my_dictionary()
list_obj = []

for line in Lines:
    
    prog_updated = []
    
    prog = line.split(" ")
    for token in prog:
    #    token = token.replace('[', '')
    #    token = token.replace(']', '')
        token = token.replace('"', '')
        token = token.replace(' ', '')
        token = token.replace(',', '')
        token = token.replace('[', '')
        token = token.replace(']', '')
        token = token.replace('\n', '')
        prog_updated.append(token)
    
   # print("prog", prog_updated)
    
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

    if(code_type in required_ctypes.keys()):
        list_obj.append({"CodeType": code_type,
                         "FeatureVector":required_ctypes[code_type],
                         "Code": prog_updated})
    
        line_count+=1
    #if(line_count>1):
     #   break
    
values, counts = np.unique(code_type_list, return_counts=True)    



with open(args.json_data_file, 'w') as json_file:
    for dict_ in list_obj:
        json.dump(dict_, json_file, 
                        separators=(',',': '))
        json_file.write('\n')
    
    
print("Total count\n", line_count)
#dict_obj = {}
#for v, c in zip(values, counts):
    
    
    #dict_obj[v] = c

    
#sorted_dict_obj = {k: v for k, v in sorted(dict_obj.items(), key=lambda item: item[1], reverse=True)[:100]}
#print("sorted_dict_obj",sorted_dict_obj) 
#print("len", len(counts))
    
    
    
    
