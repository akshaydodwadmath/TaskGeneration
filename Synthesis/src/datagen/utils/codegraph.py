# import zss
import collections
import json
import re

from src.datagen.utils.definitions import NodeType as NodeType
from src.datagen.utils.definitions import node_str_to_type as node_str_to_type

MAX_SIZE = 50
MAX_DEPTH = 10

P = 10000000007
M = 10 ** 40

MAX_POW = 40


def dsamepow10(num, max_pow=MAX_POW):  # returns the max pow of 40 in num
    r = 1
    while num >= max_pow:
        num = int(num // max_pow)
        r *= max_pow
    return r


def join_ints(a, b, max_pow=MAX_POW):
    return a * dsamepow10(b) * max_pow + b


class ASTNode:
    ''' A custom AST class to represent raw prgrams '''

    # @staticmethod
    def from_json(json_dict):
        return json_to_ast(json_dict)

    def to_int(self):
        return self._hash

    def to_json(self):
        return ast_to_json(self)

    def __init__(self, node_type_str, node_type_condition=None, children=[],
                 node_type_enum=None):

        self._type_enum = node_type_enum or node_str_to_type[node_type_str]
        self._type = node_type_str
        self._condition = node_type_condition
        self._children = children
        self._size = 0
        self._depth = 0
        self._hash = int(self._type_enum)
        if self._condition is not None:
            cond_enum = node_str_to_type[self._condition]
            self._hash = join_ints(cond_enum, self._hash, max_pow=MAX_POW)

        # counts of the constructs
        self._n_if_only = 0
        self._n_while = 0
        self._n_repeat = 0
        self._n_if_else = 0

        self._n_pick_marker = 0
        self._n_put_marker = 0
        self._n_bool_marker_present = 0
        self._n_bool_no_marker_present = 0

        for child in children:
            if child._type != 'phi':
                self._size += child._size
            self._depth = max(self._depth, child._depth)
            if child._type != 'phi':
                self._hash = join_ints(child._hash, self._hash, max_pow=MAX_POW)
            self._n_if_else += child._n_if_else
            self._n_while += child._n_while
            self._n_if_only += child._n_if_only

            self._n_bool_marker_present += child._n_bool_marker_present
            self._n_bool_no_marker_present += child._n_bool_no_marker_present

            self._n_pick_marker += child._n_pick_marker
            self._n_put_marker += child._n_put_marker

        # counts of marker booleans
        if 'bool_marker_present' in self._type:
            self._n_bool_marker_present += 1
        if 'bool_no_marker_present' in self._type:
            self._n_bool_no_marker_present += 1

        if self._type_enum == NodeType.MOVE_FORWARD:
            self._size += 1
        elif self._type_enum == NodeType.TURN_LEFT:
            self._size += 1
        elif self._type_enum == NodeType.TURN_RIGHT:
            self._size += 1
        elif self._type_enum == NodeType.PICK_MARKER:
            self._size += 1
            self._n_pick_marker += 1
        elif self._type_enum == NodeType.PUT_MARKER:
            self._size += 1
            self._n_put_marker += 1
        elif self._type_enum == NodeType.WHILE:
            self._size += 1
            self._depth += 1
            self._n_while += 1
        elif self._type_enum == NodeType.REPEAT:
            self._size += 1
            self._depth += 1
            self._n_repeat += 1
        elif self._type_enum == NodeType.IF_ELSE:
            self._size += 1
            self._depth += 1
            self._n_if_else += 1
        elif self._type_enum == NodeType.IF_ONLY:
            self._size += 1
            self._depth += 1
            self._n_if_only += 1
        elif self._type_enum == NodeType.RUN:  # empty code is of size = 0; depth = 1
            self._depth += 1
        else:
            pass

    def size(self):
        return self._size

    def depth(self):
        return self._depth

    def children(self):
        return self._children

    def conditional(self):
        return self._condition

    def n_children(self):
        return len(self._children)

    def marker_activity_flag(self):
        marker_based_nodes = self._n_put_marker + self._n_pick_marker + self._n_bool_marker_present + self._n_bool_no_marker_present
        if marker_based_nodes > 0:
            return True
        else:
            return False

    def label_print(self):
        if self._condition is not None:
            label = self._type + ' ' + self._condition
        else:
            label = self._type
        return label

    def label(self):
        return self._type

    def label_enum(self):
        return self._type_enum

    def __repr__(self, offset=''):
        cs = offset + self.label_print() + '\n'
        for child in self.children():
            cs += offset + child.__repr__(offset + '   ')
        return cs

    def __hash__(self):
        return self._hash


def check_size_and_depth(
        node: ASTNode,
        max_size=MAX_SIZE,
        max_depth=MAX_DEPTH,
):
    return node.size() <= max_size and node.depth() <= max_depth


def ast_to_json(node: ASTNode):
    ''' Converts an ASTNode into a JSON dictionary.'''

    if len(node.children()) == 0:
        if node._condition is not None:
            if node._type == 'do' or node._type == 'else':
                node_dict = {'type': node._type}
            else:
                node_dict = {'type': node._type + '(' + node._condition + ')'}
        else:
            node_dict = {'type': node._type}
        return node_dict

    if node._condition is not None:
        if node._type == 'do' or node._type == 'else':
            node_dict = {'type': node._type}
        else:
            node_dict = {'type': node._type + '(' + node._condition + ')'}
    else:
        node_dict = {'type': node._type}

    children = [ast_to_json(child) for child in node.children()]

    if children:
        node_dict['children'] = children

    return node_dict

# TODO for karel we filter boolgoal and repeatuntilgoal and for hoc we filter marker(conditionals and actions and postprocess adding markers)

# converts an AST into Karel-Benchmark format: with code in 'run' key of the dict
def ast_to_json_karelgym_format(node: ASTNode):
    """ Converts an ASTNode into a JSON dictionary in KarelGym format."""
    do_else_converter = {'do': 'ifBody', 'else': 'elseBody'}
    conditional_converter = {
        'bool_path_ahead': 'frontIsClear',
        'bool_path_left': 'leftIsClear',
        'bool_path_right': 'rightIsClear',
        'bool_marker_present': 'markersPresent',
        'bool_no_marker_present': 'noMarkersPresent',
        ###
        'bool_no_path_ahead': 'notFrontIsClear',
        'bool_no_path_left': 'notLeftIsClear',
        'bool_no_path_right': 'notRightIsClear',
        ###
        'bool_goal': 'boolGoal'  # use that
        # check if this token is ever encountered
    }

    others_converter = {
        'move': 'move',
        'turn_left': 'turnLeft',
        'turn_right': 'turnRight',
        'pick_marker': 'pickMarker',
        'put_marker': 'putMarker',
        'while': 'while',
        'repeat': 'repeat',
        'if': 'if',
        'ifelse': 'ifElse',
        'run': 'run',
        'repeat_until_goal': 'repeatUntil'
        # check if this token ever occurs in the Karel benchmark dataset
    }

    if len(node.children()) == 0:
        if node._condition is not None:
            if node._type == 'ifelse':
                do_node = node.children()[0]
                do_children = [ast_to_json_karelgym_format(child) for child in
                               do_node.children()]
                else_node = node.children()[1]
                else_children = [ast_to_json_karelgym_format(child) for child in
                                 else_node.children()]
                node_dict = {
                    'type': others_converter[node._type],
                    'condition': conditional_converter[node._condition],
                    'ifBody': do_children,
                    'elseBody': else_children
                    }
            else:
                if node._type == 'repeat':
                    node_dict = {
                                    'type': others_converter[node._type],
                                    'times': int(node._condition),
                                 }
                else:
                    node_dict = {
                            'type': others_converter[node._type],
                            'condition': conditional_converter[node._condition],
                        }
        else:
            node_dict = {'type': others_converter[node._type]}
        return node_dict

    if node._condition is not None:
        if node._type == 'ifelse':
            do_node = node.children()[0]
            do_children = [ast_to_json_karelgym_format(child) for child in
                           do_node.children()]
            else_node = node.children()[1]
            else_children = [ast_to_json_karelgym_format(child) for child in
                             else_node.children()]
            node_dict = {
                'type': others_converter[node._type],
                'condition': conditional_converter[node._condition],
                'ifBody': do_children,
                'elseBody': else_children,
                }
        else:
            if node._type == 'repeat':
                node_dict = {
                             'type': others_converter[node._type],
                             'times': int(node._condition)
                             }
            else:
                node_dict = {
                    'type': others_converter[node._type],
                    'condition': conditional_converter[node._condition]
                }
    else:
        node_dict = {'type': others_converter[node._type]}

    children = [ast_to_json_karelgym_format(child) for child in node.children() if
                (child._type != 'do' and child._type != 'else')]

    if children:
        node_dict['body'] = children

    return node_dict


def remove_null_nodes(root: ASTNode):
    queue = collections.deque([root])
    while len(queue):
        for i in range(len(queue)):
            node = queue.popleft()

            node._children = list(filter((ASTNode('phi')).__ne__, node._children))

            # for child in node._children:
            #     if child._type == 'phi':
            #         node._children.remove(child)

            for child in node._children:
                queue.append(child)

    return root


def json_to_ast(root):
    ''' Converts a JSON dictionary to an ASTNode.'''

    def get_children(json_node):
        children = json_node.get('children', [])
        return children

    node_type = root['type']
    children = get_children(root)

    if node_type == 'run':
        return ASTNode('run', None, [json_to_ast(child) for child in children])

    if '(' in node_type:
        node_type_and_cond = node_type.split('(')
        node_type_only = node_type_and_cond[0]
        cond = node_type_and_cond[1][:-1]

        if node_type_only == 'ifelse':
            assert (len(children) == 2)  # Must have do, else nodes for children

            do_node = children[0]
            assert (do_node['type'] == 'do')
            else_node = children[1]
            assert (else_node['type'] == 'else')
            do_list = [json_to_ast(child) for child in get_children(do_node)]
            else_list = [json_to_ast(child) for child in get_children(else_node)]
            # node_type is 'maze_ifElse_isPathForward' or 'maze_ifElse_isPathLeft' or 'maze_ifElse_isPathRight'
            return ASTNode(
                'ifelse', cond,
                [ASTNode('do', cond, do_list), ASTNode('else', cond, else_list)],
            )

        elif node_type_only == 'if':
            assert (len(children) == 1)  # Must have condition, do nodes for children

            do_node = children[0]
            assert (do_node['type'] == 'do')

            do_list = [json_to_ast(child) for child in get_children(do_node)]

            return ASTNode(
                'if', cond,
                do_list,
            )

        elif node_type_only == 'while':

            while_list = [json_to_ast(child) for child in children]
            return ASTNode(
                'while',
                cond,
                while_list,
            )

        elif node_type_only == 'repeat_until_goal':

            repeat_until_goal_list = [json_to_ast(child) for child in children]
            return ASTNode(
                'repeat_until_goal',
                cond,
                repeat_until_goal_list
            )

        elif node_type_only == 'repeat':
            repeat_list = [json_to_ast(child) for child in children]
            return ASTNode(
                'repeat',
                cond,
                repeat_list,
            )

        else:
            print('Unexpected node type, failing:', node_type_only)
            assert (False)

    if node_type == 'move':
        return ASTNode('move')

    if node_type == 'turn_left':
        return ASTNode('turn_left')

    if node_type == 'turn_right':
        return ASTNode('turn_right')

    if node_type == 'pick_marker':
        return ASTNode('pick_marker')

    if node_type == 'put_marker':
        return ASTNode('put_marker')

    print('Unexpected node type, failing:', node_type)
    assert (False)

    return None


def is_last_child_turn(root: ASTNode):
    children = root.children()
    if children[-1]._type == "turn_left" or children[-1]._type == "turn_right":
        return True
    else:
        return False


def get_size(root: ASTNode):
    size = 0
    for i, child in enumerate(root._children):
        if child._type != 'phi':
            size += get_size(child)

    if root._type != 'phi' and root._type != 'else' and root._type != 'do':
        size += 1

    return size


def last_node_check(sketch: ASTNode):
    # check if last node is repeat_until_goal for each child in the tree if repeat_until_goal  exists in tree
    queue = collections.deque([sketch])
    while len(queue):
        for i in range(len(queue)):
            node = queue.popleft()
            children = node.children()
            if len(children) == 0:
                continue

            elif len(children) == 1:
                if (children[
                    0]._type == 'repeat_until_goal' and node._type == 'repeat_until_goal'):
                    return False
            else:
                children_types = [c._type for c in children]
                if 'repeat_until_goal' in children_types:
                    if children_types[-1] != 'repeat_until_goal':
                        return False

            for child in node._children:
                queue.append(child)

    return True


def get_max_block_size(root: ASTNode):
    max_blk_size = 0
    node_without_children = ['move', 'turn_right', 'turn_left', 'pick_marker',
                             'put_marker']
    queue = collections.deque([root])
    while len(queue):
        for i in range(len(queue)):
            node = queue.popleft()
            node_types = [node._type for node in node._children]
            labled_nodes = ''
            for n in node_types:
                if n in node_without_children:
                    labled_nodes += 'A'
                else:

                    labled_nodes += 'O'
            # get the longest subsequence
            max_len = len(max(re.compile("(A+)*").findall(labled_nodes)))
            if max_blk_size < max_len:
                max_blk_size = max_len

            for child in node._children:
                queue.append(child)

    return max_blk_size


def extract_conditionals_from_code(root: ASTNode):
    conditional_list = []
    conditional_node_types = ['if', 'ifelse', 'while', 'repeat_until_goal']
    queue = collections.deque([root])
    while len(queue):
        for i in range(len(queue)):
            node = queue.popleft()
            node_types = [[node._type, node._condition] for node in node._children]
            for n in node_types:
                if n[0] in conditional_node_types:
                    conditional_list.append(n)

            for child in node._children:
                queue.append(child)

    return conditional_list


def valid_ASTNode(root: ASTNode):
    node_with_children = ['while', 'repeat', 'if',
                          'do', 'else', 'ifelse']
    node_repeats = ['while', 'repeat']

    only_repeats = ['repeat']

    queue = collections.deque([root])
    while len(queue):
        for i in range(len(queue)):
            node = queue.popleft()

            node_children = [n._type for n in node._children]
            if node._type in node_with_children:
                if len(node._children) == 0:
                    return False

            # to avoid cases with repeat(X1){a}; repeat(X1){a}/ while(b){c}; while(b){c}
            for i in range(len(node_children)):
                if node_children[i] in node_repeats:
                    if i + 1 < len(node_children):

                        if node_children[i + 1] == node_children[i]:
                            if (node._children[i + 1]._hash == node._children[i]._hash):
                                return False

                    if i - 1 >= 0:
                        # if node_children[i - 1] == node_children[i]:
                        #     return False
                        if node_children[i - 1] == node_children[i]:
                            if (node._children[i - 1]._hash == node._children[i]._hash):
                                return False

                # to avoid cases with repeat(X1){a}; repeat(X2){a}
                if node_children[i] in only_repeats:
                    if i + 1 < len(node_children):
                        if node_children[i + 1] in only_repeats:
                            child_1 = node._children[i]._children
                            child_2 = node._children[i + 1]._children
                            if len(child_1) == len(child_2):
                                flag = False
                                for j in range(len(child_1)):
                                    if (child_1[j]._hash != child_2[j]._hash):
                                        flag = True
                                        break
                                return flag

                    if i - 1 >= 0:
                        if node_children[i - 1] in only_repeats:
                            child_1 = node._children[i]._children
                            child_2 = node._children[i - 1]._children
                            if len(child_1) == len(child_2):
                                flag = False
                                for j in range(len(child_1)):
                                    if (child_1[j]._hash != child_2[j]._hash):
                                        flag = True
                                        break
                                return flag

            for child in node._children:
                queue.append(child)

    return True


def get_hash_code_of_ast(root: ASTNode):
    hash = int(root._type_enum)

    if root._children == [] or root._children is None:
        return hash

    for i, child in enumerate(root._children):
        if child._type != 'phi':
            hash = join_ints(get_hash_code_of_ast(child) * (i + 1), hash,
                             max_pow=MAX_POW)

    return hash


if __name__ == "__main__":
    codefile = '../../data/realworld/karel_2grids/data_format_separate/in-karel-H_code.json'

    # Convert OUR json to ASTNode
    with open(codefile, 'r') as fp:
        codeastjson = json.load(fp)
    codeast = json_to_ast(codeastjson)
    print("AST:", codeast)

    # Convert ASTNode to json in karelgym format
    codeastjson_new = ast_to_json_karelgym_format(codeast)
    codeastjson_benchmark = {}
    codeastjson_benchmark['run'] = codeastjson_new['run']
    print("Formatted Json:", codeastjson_benchmark)
