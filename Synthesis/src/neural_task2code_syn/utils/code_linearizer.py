import copy

import torch

import src.neural_task2code_syn.utils.actions as actions
from src.neural_task2code_syn.utils.vocab import tok2idx


def traverse_block(parent, relationship, location):
    block = parent[relationship]
    for st_idx, node in enumerate(block):
        child_location = location.add(relationship, st_idx)
        type = node['type']
        if {"type": type} in actions.BASIC_ACTIONS:
            print(type)
            pass
        elif type == 'cursor':
            print("cursor")
            pass
        elif type == 'repeat':
            times = node['times']
            print(f"repeat{times}")
            for i in range(1):
                traverse_block(node, 'body', child_location)
        elif type == 'while':
            print(type)
            traverse_block(node, 'body', child_location)
        elif type == 'if':
            print(type)
            traverse_block(node, 'body', child_location)
        elif type == 'ifElse':
            if node["condition"]["type"] == "not":
                print(type, node['condition']["type"],
                      node['condition']["condition"]["type"])
            else:
                print(type, node['condition']["type"])
            traverse_block(node, 'ifBody', child_location)
            traverse_block(node, 'elseBody', child_location)
        else:
            raise Exception("Unknown type: {0}".format(type))


class EmuLocationTuple:
    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __str__(self):
        return "{0}:{1}".format(self.name, self.index)


class EmuLocation:
    def __init__(self, tuples):
        self.tuples = tuples

    def add(self, name, index):
        tuples = copy.deepcopy(self.tuples)
        tuples.append(EmuLocationTuple(name, index))
        return EmuLocation(tuples)

    def __str__(self):
        return " ".join([str(x) for x in self.tuples])


def traverse_pre_order(block, teacher_code):
    if 'run' in block:
        return traverse_pre_order(block['run'], teacher_code)
    action_list = []
    for node in block:
        type = node['type']
        if {"type": type} in actions.BASIC_ACTIONS:
            action = node.copy()
            action = get_action_token(action)
            action_list.append(action)
        elif type == 'cursor':
            action = node.copy()
            action = get_action_token(action)
            action_list.append(action)

        elif type == 'repeat':
            action = node.copy()
            action['body'] = []
            action = get_action_token(action)
            action_list.append(action)
            if not teacher_code:
                action = get_action_token("REPEATBODY")
                action_list.append(action)
            action_list += traverse_pre_order(node['body'], teacher_code)

            action_list.append(get_action_token("END_BODY"))

        elif type == 'while':
            action = node.copy()
            action['body'] = []
            action = get_action_token(action)
            action_list.append(action)
            if not teacher_code:
                action = get_action_token("WHILEBODY")
                action_list.append(action)
            action_list += traverse_pre_order(node['body'], teacher_code)
            action_list.append(get_action_token("END_BODY"))

        elif type == 'if':
            action = node.copy()
            action['body'] = []
            action = get_action_token(action)
            action_list.append(action)
            if not teacher_code:
                action = get_action_token("IFBODY")
                action_list.append(action)
            action_list += traverse_pre_order(node['body'], teacher_code)
            action_list.append(get_action_token("END_BODY"))
        elif type == 'ifElse':
            action = node.copy()
            action['ifBody'] = []
            action['elseBody'] = []
            action = get_action_token(action)
            action_list.append(action)
            if not teacher_code:
                action = get_action_token("IFBODY")
                action_list.append(action)
            action_list += traverse_pre_order(node['ifBody'], teacher_code)

            action_list.append(get_action_token("END_BODY"))
            if not teacher_code:
                action = get_action_token("ELSEBODY")
                action_list.append(action)
            action_list += traverse_pre_order(node['elseBody'], teacher_code)
            action_list.append(get_action_token("END_BODY"))
        else:
            raise Exception("Unknown type: {0}".format(type))
    return action_list


# TODO remove it. It's obsolete
def extract_last_token(action_lists: torch.Tensor):
    b = action_lists == 42
    index = b.nonzero()[:, 1]
    index = torch.where(index == 0, 1, index)  # to handle edge case in the beginning
    prev_tkn = torch.diagonal(action_lists[:, index - 1], 0)

    mask = torch.isin(prev_tkn, torch.tensor([43, 44, 45, 46]))
    index = torch.where(mask, index - 2, index - 1)
    return torch.diagonal(action_lists[:, index], 0).reshape(-1, 1).int()


# We may have this in actions.py
def get_action_token(action):
    if action == actions.MOVE:
        return "MOVE"
    elif action == actions.TURN_LEFT:
        return "TURN_LEFT"
    elif action == actions.TURN_RIGHT:
        return "TURN_RIGHT"
    elif action == actions.PUT_MARKER:
        return "PUT_MARKER"
    elif action == actions.PICK_MARKER:
        return "PICK_MARKER"
    elif action == actions.WHILE_MARKERS_PRESENT:
        return "WHILE_MARKERS_PRESENT"
    elif action == actions.WHILE_NO_MARKERS_PRESENT:
        return "WHILE_NO_MARKERS_PRESENT"
    elif action == actions.WHILE_FRONT_IS_CLEAR:
        return "WHILE_FRONT_IS_CLEAR"
    elif action == actions.WHILE_NOT_FRONT_IS_CLEAR:
        return "WHILE_NOT_FRONT_IS_CLEAR"
    elif action == actions.WHILE_LEFT_IS_CLEAR:
        return "WHILE_LEFT_IS_CLEAR"
    elif action == actions.WHILE_NOT_LEFT_IS_CLEAR:
        return "WHILE_NOT_LEFT_IS_CLEAR"
    elif action == actions.WHILE_RIGHT_IS_CLEAR:
        return "WHILE_RIGHT_IS_CLEAR"
    elif action == actions.WHILE_NOT_RIGHT_IS_CLEAR:
        return "WHILE_NOT_RIGHT_IS_CLEAR"
    elif action == actions.IF_MARKERS_PRESENT:
        return "IF_MARKERS_PRESENT"
    elif action == actions.IF_NO_MARKERS_PRESENT:
        return "IF_NO_MARKERS_PRESENT"
    elif action == actions.IF_FRONT_IS_CLEAR:
        return "IF_FRONT_IS_CLEAR"
    elif action == actions.IF_NOT_FRONT_IS_CLEAR:
        return "IF_NOT_FRONT_IS_CLEAR"
    elif action == actions.IF_LEFT_IS_CLEAR:
        return "IF_LEFT_IS_CLEAR"
    elif action == actions.IF_NOT_LEFT_IS_CLEAR:
        return "IF_NOT_LEFT_IS_CLEAR"
    elif action == actions.IF_RIGHT_IS_CLEAR:
        return "IF_RIGHT_IS_CLEAR"
    elif action == actions.IF_NOT_RIGHT_IS_CLEAR:
        return "IF_NOT_RIGHT_IS_CLEAR"
    elif action == actions.IFELSE_MARKERS_PRESENT:
        return "IFELSE_MARKERS_PRESENT"
    elif action == actions.IFELSE_NO_MARKERS_PRESENT:
        return "IFELSE_NO_MARKERS_PRESENT"
    elif action == actions.IFELSE_FRONT_IS_CLEAR:
        return "IFELSE_FRONT_IS_CLEAR"
    elif action == actions.IFELSE_NOT_FRONT_IS_CLEAR:
        return "IFELSE_NOT_FRONT_IS_CLEAR"
    elif action == actions.IFELSE_LEFT_IS_CLEAR:
        return "IFELSE_LEFT_IS_CLEAR"
    elif action == actions.IFELSE_NOT_LEFT_IS_CLEAR:
        return "IFELSE_NOT_LEFT_IS_CLEAR"
    elif action == actions.IFELSE_RIGHT_IS_CLEAR:
        return "IFELSE_RIGHT_IS_CLEAR"
    elif action == actions.IFELSE_NOT_RIGHT_IS_CLEAR:
        return "IFELSE_NOT_RIGHT_IS_CLEAR"
    elif action == actions.REPEAT_1:
        return "REPEAT_1"
    elif action == actions.REPEAT_2:
        return "REPEAT_2"
    elif action == actions.REPEAT_3:
        return "REPEAT_3"
    elif action == actions.REPEAT_4:
        return "REPEAT_4"
    elif action == actions.REPEAT_5:
        return "REPEAT_5"
    elif action == actions.REPEAT_6:
        return "REPEAT_6"
    elif action == actions.REPEAT_7:
        return "REPEAT_7"
    elif action == actions.REPEAT_8:
        return "REPEAT_8"
    elif action == actions.REPEAT_9:
        return "REPEAT_9"
    elif action == actions.REPEAT_10:
        return "REPEAT_10"
    elif action == actions.REPEAT_11:
        return "REPEAT_11"
    elif action == actions.REPEAT_12:
        return "REPEAT_12"
    elif action is None or action == "END_BODY":  # action == actions.END_BODY:
        return "END_BODY"
    elif action == {"type": "cursor"}:
        return "CURSOR"
    elif action == "IFBODY":
        return "IFBODY"
    elif action == "ELSEBODY":
        return "ELSEBODY"
    elif action == "REPEATBODY":
        return "REPEATBODY"
    elif action == "WHILEBODY":
        return "WHILEBODY"
    else:
        raise Exception("Unknown action: {0}".format(action))


def translate(tokens):
    return [tok2idx[t] for t in tokens]
