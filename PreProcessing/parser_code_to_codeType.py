import numpy as np
import math
import argparse

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


required_ctypes = ['( )',
                   '( D_CONST ( ) )', 
                   '( D_CONST ( ) D_CONST ( ) D_CONST ( ) )',
                   '( D_CONST ( ) D_CONST ( D_CONST ( ) ) )',
                   '( D_CONST ( ) D_CONST ( D_CONST ( D_CONST ( ) ) ) )',
                   '( D_CONST ( ) D_CONST ( D_IF ( ) D_ELSE ( ) ) )',
                   '( D_CONST ( ) D_IF ( ) D_ELSE ( ) )',
                   '( D_CONST ( ) D_IF ( D_CONST ( ) ) D_ELSE ( ) )',
                   '( D_CONST ( D_CONST ( ) ) )',
                   '( D_CONST ( D_CONST ( ) ) D_CONST ( ) )',
                   '( D_CONST ( D_CONST ( ) ) D_CONST ( ) D_CONST ( ) )']
                   
                   

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


parser = argparse.ArgumentParser(
    description='Convert code to code types.')
add_args(parser)
args = parser.parse_args()

code_file = open(args.code_file, 'r')

code_type_file = open(args.code_type_file, "w")

Lines = code_file.readlines() 

line_count = 0
code_type_list = []
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
    constuct = 1
    ifelse_closed = False
    
    for token in prog_updated:
        if(token in commands):
            pattern_only.append("D_CONST" )
           # constuct+=1
        if(token == command_if_else[0]):
            pattern_only.append("D_" + "IF")
        if(token == command_if_else[1]):
            pattern_only.append("D_" + "ELSE")
           # constuct+=1
        if(token in control_open):
            pattern_only.append("(")
        if(token in control_close):
            pattern_only.append(")")
            
            
   # print("pattern_only", pattern_only)
    code_type = " ".join(pattern_only) 
    
    code_type_list.append(code_type)
    code_type_file.write( code_type + "\n" )

    
    line_count+=1
    #if(line_count>1):
     #   break
    
values, counts = np.unique(code_type_list, return_counts=True)    
#dict_obj = my_dictionary()
dict_obj = {}

for v, c in zip(values, counts):
    
    
    dict_obj[v] = c

    
sorted_dict_obj = {k: v for k, v in sorted(dict_obj.items(), key=lambda item: item[1], reverse=True)[:100]}
print("sorted_dict_obj",sorted_dict_obj) 
print("len", len(counts))
    
    
    
    
