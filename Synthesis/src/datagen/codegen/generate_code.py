import collections
import copy
import json
import os
import random
import time

from src.datagen.codegen.class_symast import SymAst, json_to_symast, symast_to_json
from src.datagen.wrappers.config_one_taskcode import BASIC_KAREL_ACTION_BLOCKS, BASIC_HOC_ACTION_BLOCKS, \
    OTHER_KAREL_BLOCKS, OTHER_HOC_BLOCKS

NEW_A_COUNTER = 100000
ELSE_A_COUNTER = 200000


# Util: function to generate one random partition of n, of size k
def random_partition(n, k):
    partition = [0] * k
    for x in range(n):
        partition[random.randrange(k)] += 1
    return partition


# Util to obtain an integer partition of all the A_vals in the SymAst
def get_integer_partitions(vals: dict, N: int):
    int_to_be_partitioned = N
    for key, ele in vals.items():
        if ele['minval'] is not None:
            int_to_be_partitioned -= int(ele['minval'])

    if int_to_be_partitioned < 0:
        # print("Cannot fulfill assignments: Negative integer encountered!", int_to_be_partitioned)
        return None

    # generate one random partition of size k
    rand_partition = random_partition(int_to_be_partitioned, len(vals))

    new_vals = {}
    counter = 0
    for key, ele in vals.items():
        if ele['minval'] is not None:
            ele['assigned_val'] = rand_partition[counter] + int(ele['minval'])
        else:
            ele['assigned_val'] = rand_partition[counter]
        new_vals[key] = ele
        counter += 1

    return new_vals


# fills in the D values in the tree, and obtains all the A nodes
def get_all_vals_from_symast_and_fill_assigns_D(root: SymAst, program_type: str):

    if program_type == "hoc":
        OTHER_BLOCKS = OTHER_HOC_BLOCKS
    else:
        OTHER_BLOCKS = OTHER_KAREL_BLOCKS

    IF_ELSE_DECISION_VAR = [1, 2, 3]  # 1: put children if IF-node, 2: put children in the ELSE-Node; 3: put children in both IF-ELSE Nodes
    new_A_counter = NEW_A_COUNTER
    all_vals_D = {}
    all_vals_A = {}
    counter_of_D = 0
    queue = collections.deque([root])
    while len(queue):
        for i in range(len(queue)):
            node = queue.popleft()
            if node._type == 'D' and len(node._values) == 0:
                # set the value of the D if it is not assigned
                counter_of_D += 1
                node._values = random.choices(OTHER_BLOCKS, k=1)
                if 'ifelse' in node._values[0]:
                    do_child = SymAst('do', 'DO')
                    else_child = SymAst('else', 'ELSE')
                    decision_var = random.choice(IF_ELSE_DECISION_VAR)
                    if decision_var == 1:  # Children in IF_NODE
                        do_child._children = copy.deepcopy(node._children)
                        else_child._children = [
                            SymAst('A', str(new_A_counter), minval=1)]
                        new_A_counter += 1
                    elif decision_var == 2:  # Children in ELSE_NODE
                        do_child._children = [SymAst('A', str(new_A_counter), minval=1)]
                        else_child._children = copy.deepcopy(node._children)
                        new_A_counter += 1
                    else:  # decision_var is 3
                        do_child._children = copy.deepcopy(node._children)
                        else_children = []
                        for c in do_child._children:
                            c_copy = copy.deepcopy(c)
                            # check if the id is an integer
                            c_copy_id = c_copy._id
                            if c_copy_id not in [None, 'DO', 'ELSE', 'ACTION_BLK']:
                                new_id = int(c_copy_id) + ELSE_A_COUNTER
                                c_copy._id = str(new_id)
                            else:
                                c_copy._id = c_copy_id
                            else_children.append(c_copy)

                        else_child._children = else_children
                    node._children = [do_child, else_child]
                elif 'if(' in node._values[0]:
                    do_child = SymAst('do', 'DO')
                    do_child._children = copy.deepcopy(node._children)
                    node._children = [do_child]
                else:
                    pass

                # add the value, minval attributes of the node
                node_dict = {'type': node._type, 'id': node._id, 'minval': node._minval,
                             'val': node._values}
                all_vals_D[node._type + '_' + node._id] = node_dict
            elif node._type == 'A':
                # add the value, minval attributes of the node
                node_dict = {'type': node._type, 'id': node._id, 'minval': node._minval,
                             'val': node._values}
                all_vals_A[node._type + '_' + node._id] = node_dict
            else:
                pass

            for child in node._children:
                queue.append(child)

    root._size = counter_of_D  # we could have modified the size because of the addition of another D node in the else branch
    return root, all_vals_A


# fills in the A_vals in the tree and expands those nodes
def fill_assigns_A_and_expand_symast(root: SymAst, A_vals: dict, program_type: str):

    if program_type == "hoc":
        BASIC_ACTION_BLOCKS = BASIC_HOC_ACTION_BLOCKS
    else:
        BASIC_ACTION_BLOCKS = BASIC_KAREL_ACTION_BLOCKS

    queue = collections.deque([root])
    while len(queue) > 0:
        for i in range(len(queue)):
            node = queue.popleft()
            # generate complete children for this SymAST node: expand nodes with A
            if node._children:  # in the beginning this always evaluates to true because run alsways has at least one child: A
                new_children = []
                for ele in node._children:
                    if ele._type == 'A':
                        node_name = ele._type + '_' + ele._id
                        ele._values = random.choices(BASIC_ACTION_BLOCKS,
                                                     k=A_vals[node_name][
                                                         'assigned_val'])
                        for new_child in ele._values:
                            new_children.append(SymAst(new_child, 'ACTION_BLK'))
                    else:
                        new_children.append(ele)
                node._children = new_children

            for child in node._children:
                queue.append(child)

    return root


# Main function: that generates one code of size max_blocks, given code type and max_blocks
def generate_one_code(code_type: dict, num_blocks: int, program_type: str):
    # convert the JSON code type into SymAst
    symast_code = json_to_symast(code_type)

    # obtain all the values of the nodes, and fill in D
    # we might miss some codes (symast size and minval size are variable and can be incompatible)
    symast_code, all_vals_A = get_all_vals_from_symast_and_fill_assigns_D(symast_code, program_type=program_type)
    available_blocks = num_blocks - symast_code.size()

    # also check the minval requirement of each of the A nodes in the Tree, and if it can be satisfied by the available blocks
    minval_counter = []
    for key, val in all_vals_A.items():
        if val['minval'] is not None:
            minval_counter.append(int(val['minval']))
    minval_size = sum(minval_counter)
    satisfiable_blocks = available_blocks - minval_size
    if satisfiable_blocks <= 0:
        # print("Cannot fulfill assignments: Negative integer encountered!", satisfiable_blocks)
        return None, False

    # get the integer partitions
    int_partitions_A = get_integer_partitions(all_vals_A, available_blocks)
    # fill in the assignment for A and expand the SymAst node
    symast_expanded = fill_assigns_A_and_expand_symast(symast_code, int_partitions_A, program_type=program_type)

    # check the children of run, and assign ifelse flag
    ifelse_flag = False
    children_types = []
    for c in symast_expanded.children():
        if c._type == 'D':
            children_types.append(c._values[0].split('(')[0])
        else:
            children_types.append(c._type)
    if 'ifelse' in children_types:
        ifelse_flag = True
    # convert SymAst into JSON object
    symast_json = symast_to_json(symast_expanded)

    return symast_json, ifelse_flag


if __name__ == "__main__":

    type = "hoc"

    basic_code_type = {'type': 'run',
                       'children': [
                           {'type': 'A_1'},
                           {'type': 'D_1',
                            'children': [
                                {'type': 'A_2'},
                                {'type': 'D_2',
                                 'children': [
                                     {'type': 'A_3'},
                                     {'type': 'D_3',
                                      'children': [
                                          {'type': 'A_4', 'minval': '1'}
                                      ]},
                                     {'type': 'A_5'}
                                 ]},
                                {'type': 'A_6'}
                            ]},
                           {'type': 'A_7'}
                       ]}
    print("JSON CODE TYPE:", basic_code_type)

    N = 1
    S = 20
    save_json = False

    start_time = time.time()
    for i in range(N):
        single_code, _ = generate_one_code(basic_code_type, S, program_type=type)
        print(single_code)
        # save the codes in a file
        if save_json:
            filename = 'code_' + str(i) + '.json'
            # create the folder if it does not exist
            if not os.path.exists('../../tests/test_datasets/'):
                os.makedirs('../../tests/test_datasets/')
            with open('../../tests/test_datasets/' + filename, 'w') as fp:
                json.dump(single_code, fp, indent=4)
    end_time = time.time()

    print(
        "Time taken to generate %d code(s) of Depth = 4, Size = %d (in seconds): %f" % (
        N, S, round(end_time - start_time, 2)))
