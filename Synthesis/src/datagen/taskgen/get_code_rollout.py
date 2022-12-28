import collections
import copy
import random


class RAst:
    def __init__(self, node_type_str, id, children=[], val=[]):
        self._type = node_type_str
        self._id = id
        self._boolean_var = self.get_boolean_var(node_type_str)
        self._repeatX = self.get_repeat_var(node_type_str)
        self._values = val
        self._children = children
        self._size = 0
        self._depth = 0

        # counts of the constructs
        self._nwhile = 0
        self._nrepeat = 0
        self._nifelse = 0
        self._nifonly = 0

        for child in children:
            self._size += child._size
            self._depth = max(self._depth, child._depth)
            self._nwhile += child._nwhile
            self._nrepeat += child._nrepeat
            self._nifelse += child._nifelse
            self._nifonly += child._nifonly

        if self._type == 'do':
            pass
        if self._type == 'else':
            pass
        if 'while' in self._type:
            self._depth += 1
            self._size += 1
            self._nwhile += 1
        if 'ifelse' in self._type:
            self._depth += 1
            self._size += 1
            self._nifelse += 1
        if 'if(' in self._type:
            self._depth += 1
            self._size += 1
            self._nifonly += 1
        if 'repeat' in self._type:
            self._depth += 1
            self._size += 1
            self._nrepeat += 1
        elif self._type == 'run':  # empty code is of size = 0; depth = 1
            self._depth += 1
        else:  # for all other action nodes: only the size increases but not the depth
            self._size += 1

    def get_boolean_var(self, node_str):
        split_str = node_str.split('(')
        if len(split_str) == 1:  # no boolean param
            return None
        if 'repeat' in split_str[0]:
            return None
        else:
            boolean_var = split_str[1].split(')')[0]
            return boolean_var

    def get_repeat_var(self, node_str):
        split_str = node_str.split('(')
        if len(split_str) == 1:  # no boolean param
            return None
        if 'if' in split_str[0]:
            return None
        if 'while' in split_str[0]:
            return None
        else:
            repeat_var = split_str[1].split(')')[0]
            return repeat_var

    def size(self):
        return self._size

    def depth(self):
        return self._depth

    def children(self):
        return self._children

    def label_print(self):
        label = self._type + ' id:' + self._id + ' vals:' + str(self._values)
        return label

    def __repr__(self, offset=''):
        cs = offset + self.label_print() + '\n'
        for child in self.children():
            cs += offset + child.__repr__(offset + '   ')
        return cs


def json_to_RAst(root: dict, counter=1):
    ''' Converts a JSON dictionary to RAst Node'''

    def get_children(json_node):
        children = json_node.get('children', [])
        return children

    node_type = root['type']
    children = get_children(root)

    if node_type == 'run':
        return RAst('run', str(counter),
                    [json_to_RAst(child, counter=counter + i + 1) for i, child in
                     enumerate(children)], val=[])

    if '(' in node_type:
        node_type_and_cond = node_type.split('(')
        node_type_only = node_type_and_cond[0]

        if node_type_only == 'ifelse':
            count = counter + 1
            assert (len(children) == 2)  # Must have do, else nodes for children

            do_node = children[0]
            assert (do_node['type'] == 'do')
            else_node = children[1]
            assert (else_node['type'] == 'else')
            do_list = [json_to_RAst(child, counter=count + 1 + i) for i, child in
                       enumerate(get_children(do_node))]
            else_list = [json_to_RAst(child, counter=count + 1 + i) for i, child in
                         enumerate(get_children(else_node))]
            return RAst(
                node_type, str(count),
                [RAst('do', str(count), do_list, val=[]),
                 RAst('else', str(count), else_list, val=[])],
                val=[]
            )

        elif node_type_only == 'if':
            count = counter + 1
            assert (len(children) == 1)  # Must have condition, do nodes for children

            do_node = children[0]
            assert (do_node['type'] == 'do')

            do_list = [json_to_RAst(child, count + 1 + i) for i, child in
                       enumerate(get_children(do_node))]

            return RAst(
                node_type,
                str(count),
                do_list,
                val=[]
            )


        elif node_type_only == 'while':
            count = counter + 1
            while_list = [json_to_RAst(child, count + 1 + i) for i, child in
                          enumerate(children)]
            return RAst(
                node_type,
                str(count),
                while_list,
                val=[]
            )

        elif node_type_only == 'repeat':
            count = counter + 1
            repeat_list = [json_to_RAst(child, count + i + 1) for i, child in
                           enumerate(children)]
            return RAst(
                node_type,
                str(count),
                repeat_list,
                val=[]
            )

        elif node_type_only == 'repeat_until_goal':
            count = counter + 1
            repeat_until_list = [json_to_RAst(child, count + i + 1) for i, child in
                           enumerate(children)]
            return RAst(
                node_type,
                str(count),
                repeat_until_list,
                val=[]
            )

        else:
            print('Unexpected node type, failing:', node_type_only)
            assert (False)

    if node_type == 'move':
        return RAst('move', str(counter + 1), val=[])

    if node_type == 'turn_left':
        return RAst('turn_left', str(counter + 1), val=[])

    if node_type == 'turn_right':
        return RAst('turn_right', str(counter + 1), val=[])

    if node_type == 'pick_marker':
        return RAst('pick_marker', str(counter + 1), val=[])

    if node_type == 'put_marker':
        return RAst('put_marker', str(counter + 1), val=[])

    print('Unexpected node type, failing:', node_type)
    assert (False)

    return None


def unroll_ast(root: RAst, while_counter_min: int, while_counter_max: int):
    # first unroll the tree to get rid of all the while, repeat, if, if_else nodes
    queue = collections.deque([root])
    while len(queue) > 0:
        for i in range(len(queue)):
            node = queue.popleft()
            # generate new children for this node by expanding on the programming construct nodes
            new_children = []
            for child in node._children:
                if 'while' in child._type:
                    if len(child._values) == 0:
                        # randomly assign an unrolling value
                        child._values.append(
                            random.randint(while_counter_min, while_counter_max))
                    counter = child._values[0]
                    id = child._id
                    bool_var = child._boolean_var
                    while_children = []
                    for i in range(counter):
                        while_children.append(
                            RAst(str(bool_var) + ':true', id, val=["BOOL"]))
                        while_children.extend(copy.deepcopy(child._children))
                    for c in while_children:
                        if 'while' in c._type:
                            c._values = [
                                random.randint(while_counter_min, while_counter_max)]
                        elif 'repeat_until_goal' in c._type:
                            c._values = [
                                random.randint(while_counter_min, while_counter_max)]
                        elif 'if' in c._type:
                            c._values = [random.randint(0, 1)]
                        elif 'bool' in c._type:
                            pass
                        else:
                            c._values = []
                    while_children.append(
                        RAst(str(bool_var) + ':false', id, val=["BOOL"]))
                    new_children.extend(while_children)

                elif 'ifelse' in child._type:
                    if len(child._values) == 0:
                        # randomly assign an unrolling value
                        child._values.append(random.randint(0, 1))
                    counter = child._values[0]  # going to be 1/0
                    id = child._id
                    bool_var = child._boolean_var
                    ifelse_children = []
                    if counter:  # DO node
                        do_children = copy.deepcopy(child._children[0]._children)
                        ifelse_children.append(
                            RAst(str(bool_var) + ':true', id, val=["BOOL"]))
                        ifelse_children.extend(do_children)
                    else:  # ELSE node
                        else_children = copy.deepcopy(child._children[1]._children)
                        ifelse_children.append(
                            RAst(str(bool_var) + ':false', id, val=["BOOL"]))
                        ifelse_children.extend(else_children)
                    for c in ifelse_children:
                        if 'while' in c._type:
                            c._values = [
                                random.randint(while_counter_min, while_counter_max)]
                        elif 'repeat_until_goal' in c._type:
                            c._values = [
                                random.randint(while_counter_min, while_counter_max)]
                        elif 'if' in c._type:
                            c._values = [random.randint(0, 1)]
                        elif 'bool' in c._type:
                            pass
                        else:
                            c._values = []
                    new_children.extend(ifelse_children)

                elif 'if(' in child._type:
                    if len(child._values) == 0:
                        # randomly assign an unrolling value
                        child._values.append(random.randint(0, 1))
                    counter = child._values[0]  # going to be 1/0
                    id = child._id
                    bool_var = child._boolean_var
                    ifchildren = []
                    if counter:  # DO node
                        do_children = copy.deepcopy(child._children[0]._children)
                        ifchildren.append(
                            RAst(str(bool_var) + ':true', id, val=["BOOL"]))
                        ifchildren.extend(do_children)
                    else:  # don't add anything
                        ifchildren.append(
                            RAst(str(bool_var) + ':false', id, val=["BOOL"]))
                    for c in ifchildren:
                        if 'while' in c._type:
                            c._values = [
                                random.randint(while_counter_min, while_counter_max)]
                        elif 'repeat_until_goal' in c._type:
                            c._values = [
                                random.randint(while_counter_min, while_counter_max)]
                        elif 'if' in c._type:
                            c._values = [random.randint(0, 1)]
                        elif 'bool' in c._type:
                            pass
                        else:
                            c._values = []
                    new_children.extend(ifchildren)

                elif 'repeat_until_goal' in child._type:
                    if len(child._values) == 0:
                        # randomly assign an unrolling value
                        child._values.append(
                            random.randint(while_counter_min, while_counter_max))
                    counter = child._values[0]
                    id = child._id
                    bool_var = child._boolean_var
                    repeatuntil_children = []
                    for i in range(counter):
                        repeatuntil_children.append(
                            RAst(str(bool_var) + ':true', id, val=["BOOL"]))
                        repeatuntil_children.extend(copy.deepcopy(child._children))
                    for c in repeatuntil_children:
                        if 'while' in c._type:
                            c._values = [
                                random.randint(while_counter_min, while_counter_max)]
                        elif 'repeat_until_goal' in c._type:
                            c._values = [
                                random.randint(while_counter_min, while_counter_max)]
                        elif 'if' in c._type:
                            c._values = [random.randint(0, 1)]
                        elif 'bool' in c._type:
                            pass
                        else:
                            c._values = []
                    repeatuntil_children.append(
                        RAst(str(bool_var) + ':false', id, val=["BOOL"]))
                    new_children.extend(repeatuntil_children)

                elif 'repeat' in child._type:
                    counter = int(child._repeatX)
                    repeat_children = []
                    for i in range(counter):
                        repeat_children.extend(copy.deepcopy(child._children))
                    for c in repeat_children:
                        if 'while' in c._type:
                            c._values = [
                                random.randint(while_counter_min, while_counter_max)]
                        elif 'repeat_until_goal' in c._type:
                            c._values = [
                                random.randint(while_counter_min, while_counter_max)]
                        elif 'if(' in c._type:
                            c._values = [random.randint(0, 1)]
                        elif 'ifelse' in c._type:
                            c._values = [random.randint(0, 1)]
                        elif 'bool' in c._type:
                            pass
                        else:
                            c._values = []
                    new_children.extend(repeat_children)

                else:  # for all other cases just add the child node as is
                    new_children.append(copy.deepcopy(child))

            node._children = new_children

            # check if there are any elements in the children which need to be unrolled further:
            # if yes, add it to the queue
            children_types = [c._type.split('(')[0] for c in node._children]
            if 'while' in children_types:
                queue.append(node)
            elif 'repeat_until_goal' in children_types:
                queue.append(node)
            elif 'repeat' in children_types:
                queue.append(node)
            elif 'ifelse' in children_types:
                queue.append(node)
            elif 'if' in children_types:
                queue.append(node)
            else:
                pass

    return root


def get_rollout_tokens(root: RAst):
    code_tokens = []
    queue = collections.deque([root])
    while len(queue) > 0:
        for i in range(len(queue)):
            node = queue.popleft()
            code_tokens.append(node._type)
            for child in node._children:
                queue.append(child)

    return code_tokens


def generate_rollout(code_json: dict, while_counter_min: int, while_counter_max: int):
    # get the RAst object
    code_ast = json_to_RAst(code_json)
    # unroll the ast
    unrolled_ast = unroll_ast(code_ast, while_counter_min, while_counter_max)
    # obtain the code tokens
    unrolled_tokens = get_rollout_tokens(unrolled_ast)
    return unrolled_tokens


if __name__ == "__main__":
    example_depth1 = RAst('run', 'RUN', children=[

        RAst('move', '1'),
        RAst('turn_left', '2'),
        RAst('move', '3'),
        RAst('move', '4'),
        RAst('turn_right', '5')

    ])

    example_depth2_a = RAst('run', 'RUN', children=[
        RAst('while(bool_path_ahead)', '1', children=[
            RAst('move', '2'),
            RAst('turn_left', '3'),
            RAst('move', '4'),
            RAst('move', '5'),
        ]),
        RAst('turn_right', '6')
    ])

    example_depth2_b = RAst('run', 'RUN', children=[
        RAst('move', '7'),
        RAst('if(bool_path_left)', '1', children=[
            RAst('do', '1', children=[
                RAst('move', '2'),
                RAst('turn_left', '3'),
                RAst('move', '4')
            ])
        ]),
        RAst('move', '5'),
        RAst('turn_right', '6')
    ])

    example_depth2_c = RAst('run', 'RUN', children=[
        RAst('move', '1'),
        RAst('move', '2'),
        RAst('ifelse(bool_path_left)', '3', children=[
            RAst('do', '3', children=[
                RAst('turn_left', '4'),
                RAst('move', '5')
            ]),
            RAst('else', '3', children=[
                RAst('turn_right', '6'),
                RAst('move', '7'),
                RAst('move', '8'),
            ])
        ]),
        RAst('move', '9'),
        RAst('move', '10')
    ])

    example_depth2_d = RAst('run', 'RUN', children=[
        RAst('move', '1'),
        RAst('repeat(9)', '2', children=[
            RAst('move', '3'),
            RAst('move', '4'),
            RAst('turn_right', '5')
        ]),
        RAst('turn_left', '6')
    ])

    example_depth2_e = RAst('run', 'RUN', children=[
        RAst('move', '1'),
        RAst('while(bool_marker_present)', '2', children=[
            RAst('move', '3'),
            RAst('turn_left', '4'),
            RAst('move', '5')
        ]),
        RAst('turn_right', '6'),
        RAst('if(bool_path_ahead)', '7', children=[
            RAst('do', '7', children=[
                RAst('move', '8'),
                RAst('move', '9')
            ])
        ]),
        RAst('turn_left', '10'),
        RAst('move', '11')
    ])

    example_depth2_f = RAst('run', 'RUN', children=[
        RAst('repeat(5)', '1', children=[
            RAst('move', '2'),
            RAst('move', '3')
        ]),
        RAst('turn_left', '4'),
        RAst('move', '5'),
        RAst('ifelse(bool_path_right)', '6', children=[
            RAst('do', '7', children=[
                RAst('move', '8'),
                RAst('turn_left', '9')
            ]),
            RAst('else', '8', children=[
                RAst('move', '10'),
                RAst('move', '11')
            ])
        ]),
        RAst('move', '12')
    ])

    example_depth2_g = RAst('run', 'RUN', children=[
        RAst('while(bool_path_ahead)', '1', children=[
            RAst('while(bool_marker_present)', '2', children=[
                RAst('move', '3')
            ]),
            RAst('move', '4')
        ])
    ])

    example_depth3_a = RAst('run', 'RUN', children=[
        RAst('while(bool_no_marker_present)', '1', children=[
            RAst('move', '2'),
            RAst('ifelse(bool_path_ahead)', '3', children=[
                RAst('do', '3', children=[
                    RAst('move', '4'),
                    RAst('turn_left', '5')
                ]),
                RAst('else', '4', children=[
                    RAst('turn_right', '6'),
                    RAst('move', '7')
                ])
            ])
        ]),
        RAst('move', '8')
    ])

    example_depth3_b = RAst('run', 'RUN', children=[
        RAst('while(bool_path_ahead)', '1', children=[
            RAst('move', '2'),
            RAst('while(bool_path_right)', '3', children=[
                RAst('move', '4'),
                RAst('turn_right', '5'),
                RAst('move', '6')
            ])
        ]),
        RAst('pick_marker', '15'),
        RAst('put_marker', '16'),
        RAst('ifelse(bool_path_left)', '7', children=[
            RAst('do', '7', children=[
                RAst('move', '8'),
                RAst('if(bool_marker_present)', '9', children=[
                    RAst('do', '9', children=[
                        RAst('turn_right', '11'),
                        RAst('turn_left', '12')
                    ])
                ])
            ]),
            RAst('else', '7', children=[
                RAst('move', '18')
            ])
        ]),
        RAst('move', '13'),
        RAst('turn_left', '14')
    ])

    example_depth4 = RAst('run', 'RUN', children=[
        RAst('while(bool_no_marker_present)', '1', children=[
            RAst('move', '2'),
            RAst('ifelse(bool_path_ahead)', '3', children=[
                RAst('do', '3', children=[
                    RAst('if(bool_marker_present)', '5', children=[
                        RAst('do', '5', children=[
                            RAst('pick_marker', '6'),
                            RAst('move', '7')
                        ])
                    ])
                ]),
                RAst('else', '3', children=[
                    RAst('move', '8'),
                    RAst('move', '9'),
                    RAst('while(bool_path_ahead)', '10', children=[
                        RAst('move', '11'),
                        RAst('move', '12')
                    ])
                ])
            ])
        ]),
        RAst('turn_right', '13'),
        RAst('put_marker', '14')
    ])

    unrolled_ast = unroll_ast(example_depth4, 2, 5)
    unrolled_tokens = get_rollout_tokens(unrolled_ast)
    print("After unrolling:", len(unrolled_tokens))

    # With JSON objects
    example_depth2_json = {'type': 'run',
                           'children': [
                               {'type': 'while(bool_path_ahead)',
                                'children': [
                                    {'type': 'move'},
                                    {'type': 'put_marker'}
                                ]}

                           ]}

    # # obtain the RAst object
    # example_depth2_ast = json_to_RAst(example_depth2_json)
    # print("From JSON to AST:", example_depth2_ast)
    # example_depth2_ast_copy = copy.deepcopy(example_depth2_ast)
    # unrolled_ast = unroll_ast(example_depth2_ast, 2, 10)
    # unrolled_tokens = get_rollout_tokens(unrolled_ast)
    # print("After unrolling JSON code:", len(unrolled_tokens))
    # 
    # # example_depth2_ast = json_to_RAst(example_depth2_json)
    # unrolled_ast_copy = unroll_ast(example_depth2_ast_copy, 2, 10)
    # unrolled_tokens_copy = get_rollout_tokens(unrolled_ast_copy)
    # print("After unrolling JSON code copy:", len(unrolled_tokens_copy))

    code_json = {'type': 'run', 'children': [{'type': 'move'},
                                             {'type': 'put_marker'},
                                             {'type': 'pick_marker'},
                                             {'type': 'turn_right'},
                                             {'type': 'ifelse(bool_no_path_left)',
                                              'children': [{'type': 'do', 'children': [
                                                  {'type': 'turn_right'},
                                                  {'type': 'turn_left'},
                                                  {'type': 'turn_right'},
                                                  {'type': 'turn_right'},
                                                  {'type': 'pick_marker'}]},
                                                           {'type': 'else',
                                                            'children': [
                                                                {'type': 'put_marker'},
                                                                {'type': 'turn_right'},
                                                                {'type': 'turn_left'},
                                                                {'type': 'pick_marker'},
                                                                {'type': 'turn_left'},
                                                                {'type': 'turn_right'},
                                                                {'type': 'turn_left'},
                                                                {'type': 'put_marker'},
                                                                {'type': 'pick_marker'},
                                                                {'type': 'move'},
                                                                {'type': 'put_marker'},
                                                                {'type': 'turn_right'},
                                                                {'type': 'pick_marker'},
                                                                {
                                                                    'type': 'pick_marker'}]}]},
                                             {'type': 'put_marker'}, {'type': 'move'},
                                             {'type': 'turn_right'},
                                             {'type': 'turn_right'}, {'type': 'move'},
                                             {'type': 'put_marker'}]}

    unrolled_tokens = generate_rollout(code_json, while_counter_min=2,
                                       while_counter_max=5)
    print("Unrolled tokens:", unrolled_tokens)
