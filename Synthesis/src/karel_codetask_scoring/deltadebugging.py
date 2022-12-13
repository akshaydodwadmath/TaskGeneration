import json

from src.karel_emulator.code import Code
from src.karel_emulator.executor import Executor
from src.karel_emulator.task import Task
from src.karel_emulator.tokens import actions, simple_control, control, complex_control

# INVALID_SEQUENCES = [
#     ['turnRight', 'turnLeft'],
#     ['turnLeft', 'turnRight'],
#     ['pickMarker', 'putMarker'],
#     ['putMarker', 'pickMarker'],
#     ['turnLeft', 'pickMarker', 'turnRight'],
#     ['turnRight', 'pickMarker', 'turnLeft'],
#     ['turnLeft', 'putMarker', 'turnRight'],
#     ['turnRight', 'putMarker', 'turnLeft'],
#     ['pickMarker', 'turnLeft', 'putMarker'],
#     ['pickMarker', 'turnRight', 'putMarker'],
#     ['putMarker', 'turnLeft', 'pickMarker'],
#     ['putMarker', 'turnRight', 'pickMarker'],
#     ['turnRight', 'turnRight', 'turnRight'],
#     ['turnLeft', 'turnLeft', 'turnLeft']
# ]

# def contains(childlist, parentlist):
#     """
#
#     :param childlist:
#     :param parentlist:
#     :return: True if childlist is in parentlist; False otherwise
#     """
#     for i in range(len(parentlist) - len(childlist) + 1):
#         for j in range(len(childlist)):
#             if parentlist[i + j] != childlist[j]:
#                 break
#         else:
#             return True
#     return False

# def get_code_validity(root: ASTNode):
#     """
#
#     :param root:
#     :return: True if code IS VALID; False if code is INVALID
#     """
#     validity_flag = True
#     queue = collections.deque([root])
#     while len(queue) > 0:
#         for i in range(len(queue)):
#             node = queue.popleft()
#             # obtain the types of the children
#             if node._children:
#                 children_types = [c._type for c in node._children]
#                 for ele in INVALID_SEQUENCES:
#                     if len(children_types) >= len(ele):
#                         flag = contains(ele, children_types)
#                         if flag:  # invalid seq in the children_types
#                             validity_flag = False
#                             return validity_flag  # code NOT valid
#
#             for child in node._children:
#                 queue.append(child)
#
#     return validity_flag
#
# def check_code_redundancy(code_json: dict):
#     code_ast = json_to_ast(code_json)
#     flag = get_code_validity(code_ast)
#
#     return flag

INVALID_SEQUENCES = [
    # Pointless simple action sequences
    [{'type': 'turnRight'}, {'type': 'turnLeft'}],
    [{'type': 'turnLeft'}, {'type': 'turnRight'}],
    [{'type': 'pickMarker'}, {'type': 'putMarker'}],
    [{'type': 'putMarker'}, {'type': 'pickMarker'}],
    [{'type': 'turnLeft'}, {'type': 'pickMarker'}, {'type': 'turnRight'}],
    [{'type': 'turnRight'}, {'type': 'pickMarker'}, {'type': 'turnLeft'}],
    [{'type': 'turnLeft'}, {'type': 'putMarker'}, {'type': 'turnRight'}],
    [{'type': 'turnRight'}, {'type': 'putMarker'}, {'type': 'turnLeft'}],
    [{'type': 'pickMarker'}, {'type': 'turnLeft'}, {'type': 'putMarker'}],
    [{'type': 'pickMarker'}, {'type': 'turnRight'}, {'type': 'putMarker'}],
    [{'type': 'putMarker'}, {'type': 'turnLeft'}, {'type': 'pickMarker'}],
    [{'type': 'putMarker'}, {'type': 'turnRight'}, {'type': 'pickMarker'}],
    [{'type': 'turnRight'}, {'type': 'turnRight'}, {'type': 'turnRight'}],
    [{'type': 'turnLeft'}, {'type': 'turnLeft'}, {'type': 'turnLeft'}],
    # Useless loops
    [{'type': 'repeat', 'times': 2, 'body': [{'type': 'turnLeft'},
                                             {'type': 'turnLeft'}]}],
    [{'type': 'repeat', 'times': 2, 'body': [{'type': 'turnRight'},
                                             {'type': 'turnRight'}]}],
    [{'type': 'repeat', 'times': 4, 'body': [{'type': 'turnLeft'}]}],
    [{'type': 'repeat', 'times': 4, 'body': [{'type': 'turnRight'}]}],
    [{'type': 'repeat', 'times': 5, 'body': [{'type': 'turnRight'}]}],
    [{'type': 'repeat', 'times': 5, 'body': [{'type': 'turnLeft'}]}],
    # Borderline useless
    # [{'type': 'repeat', 'times': 5, 'body': [{'type': 'turnRight'},
    #                                          {'type': 'pickMarker'}]}],
    # [{'type': 'repeat', 'times': 5, 'body': [{'type': 'turnRight'},
    #                                          {'type': 'putMarker'}]}],
    # [{'type': 'repeat', 'times': 5, 'body': [{'type': 'turnLeft'},
    #                                          {'type': 'pickMarker'}]}],
    # [{'type': 'repeat', 'times': 5, 'body': [{'type': 'turnLeft'},
    #                                          {'type': 'putMarker'}]}],
]

negate = {
    'frontIsClear': 'notFrontIsClear',
    'leftIsClear': 'notLeftIsClear',
    'rightIsClear': 'notRightIsClear',
    'notFrontIsClear': 'frontIsClear',
    'notLeftIsClear': 'leftIsClear',
    'notRightIsClear': 'rightIsClear',
    'markersPresent': 'noMarkersPresent',
    'noMarkersPresent': 'markersPresent',
}


def contains_sublist(lst, sublst):
    """
    :param lst: list to be checked
    :param sublst: sublist to be checked
    :return: True if sublst is in lst; False otherwise
    """
    n = len(sublst)
    return any((sublst == lst[i:i + n]) for i in range(len(lst) - n + 1))


def check_body_redundancy(body: list):
    """
    :param body: list of actions
    :return: True if body is valid; False otherwise
    """
    for action in body:
        if action['type'] in simple_control:
            inner_validity = check_body_redundancy(action['body'])
            if not inner_validity:
                return False
        elif action['type'] in complex_control:
            inner_validity = check_body_redundancy(action['ifBody'])
            if not inner_validity:
                return False
            inner_validity = check_body_redundancy(action['elseBody'])
            if not inner_validity:
                return False
    for el in INVALID_SEQUENCES:
        if contains_sublist(body, el):
            return False
    return True


def check_code_redundancy(code: Code):
    """
    :param code: code to be checked
    :return: True if code is valid; False otherwise
    """
    body = code.astJson['body']
    return check_body_redundancy(body)


def remove_basic_action(body: list, action: dict, index: int):
    """
    :param body: list of actions
    :param action: action to be removed
    :param index: index of the action to be removed
    """
    new_body = []
    for i, ele in enumerate(body):
        if i == index:
            continue
        new_body.append(ele)
    return new_body


def simple_inside_remove(body: list, action: dict, index: int,
                         keep_empty_body=False):
    """
    Method that handles while, if and repeat
    :param body: list of actions
    :param action: action to be removed
    :param index: index of the action to be removed
    :param keep_empty_body: if True, then empty bodies will also be present in the
    final list
    :return: list of delta bodies
    """
    new_body = []
    for i, el in enumerate(body):
        if i == index:
            continue
        new_body.append(el)
    new_bodies = []
    list_of_bodies = get_list_of_delta_bodies(action['body'], keep_empty_body)
    if not any(list_of_bodies) and not keep_empty_body:
        return [new_body]
    for el in list_of_bodies:
        if action['type'] == 'repeat':
            new = [{'type': action['type'],
                    'times': action['times'],
                    'body': el}]
        else:
            new = [{'type': action['type'],
                    'condition': action['condition'],
                    'body': el}]
        new_bodies.append(new_body[:index] +
                          new +
                          new_body[index:])

    return new_bodies


def do_unwrap(body: list, action: dict, index: int):
    """
    :param body: list of actions
    :param action: action to be removed
    :param index: index of the action to be removed
    """
    new_body = []
    for i, el in enumerate(body):
        if i == index:
            pass
        else:
            new_body.append(el)
    new_bodies = []
    if action['type'] in simple_control:
        new_bodies.append(new_body[:index] + action['body'] + new_body[index:])
        for i, el in enumerate(action['body']):
            if el['type'] in control:
                for el2 in do_unwrap(action['body'], el, i):
                    # new_bodies.append(new_body[:index] + el2 + new_body[index:])
                    if action['type'] == 'repeat':
                        new = new_body[:index] + \
                              [{'type': action['type'],
                                'times': action['times'],
                                'body': el2}] + \
                              new_body[index:]
                    else:
                        new = new_body[:index] + \
                              [{'type': action['type'],
                                'condition': action['condition'],
                                'body': el2}] + \
                              new_body[index:]
                    new_bodies.append(new)
    elif action['type'] == 'ifElse':
        new_bodies.append(new_body[:index] + action['ifBody'] +
                          action['elseBody'] + new_body[index:])
        for i, el in enumerate(action['ifBody']):
            if el['type'] in control:
                for el2 in do_unwrap(action['ifBody'], el, i):
                    # new_bodies.append(new_body[:index] + el2 + new_body[index:])
                    if action['type'] == 'repeat':
                        new = new_body[:index] + \
                              [{'type': action['type'],
                                'times': action['times'],
                                'ifBody': el2,
                                'elseBody': action['elseBody']}] + \
                              new_body[index:]
                    else:
                        new = new_body[:index] + \
                              [{'type': action['type'],
                                'condition': action['condition'],
                                'ifBody': el2,
                                'elseBody': action['elseBody']}] + \
                              new_body[index:]
                    new_bodies.append(new)
        for i, el in enumerate(action['elseBody']):
            if el['type'] in control:
                for el2 in do_unwrap(action['elseBody'], el, i):
                    # new_bodies.append(new_body[:index] + el2 + new_body[index:])
                    if action['type'] == 'repeat':
                        new = new_body[:index] + \
                              [{'type': action['type'],
                                'times': action['times'],
                                'ifBody': action['ifBody'],
                                'elseBody': el2}] + \
                              new_body[index:]
                    else:
                        new = new_body[:index] + \
                              [{'type': action['type'],
                                'condition': action['condition'],
                                'ifBody': action['ifBody'],
                                'elseBody': el2}] + \
                              new_body[index:]
                    new_bodies.append(new)
    return new_bodies


def complex_inside_remove(body: list, action: dict, index: int, keep_empty_body=False):
    """
    :param body: list of actions
    :param action: action to be removed
    :param index: index of the action to be removed
    :param keep_empty_body: if True, then empty bodies will also be present in the
    final list
    :return: list of delta bodies
    """
    new_body = []
    for i, el in enumerate(body):
        if i == index:
            continue
        new_body.append(el)
    new_bodies = []
    list_of_bodies = get_list_of_delta_bodies(action['ifBody'], keep_empty_body)
    if not any(list_of_bodies) and not keep_empty_body:
        new = [{'type': 'if',
                'condition': negate[action['condition']],
                'body': action['elseBody']}]
        new_bodies.append(new_body[:index] +
                          new +
                          new_body[index:])
    else:
        for el in list_of_bodies:
            new = [{'type': action['type'],
                    'condition': action['condition'],
                    'ifBody': el,
                    'elseBody': action['elseBody']}]
            new_bodies.append(new_body[:index] +
                              new +
                              new_body[index:])
    list_of_bodies = get_list_of_delta_bodies(action['elseBody'], keep_empty_body)
    if not any(list_of_bodies) and not keep_empty_body:
        new = [{'type': 'if',
                'condition': action['condition'],
                'body': action['ifBody']}]
        new_bodies.append(new_body[:index] +
                          new +
                          new_body[index:])
    else:
        for el in list_of_bodies:
            new = [{'type': action['type'],
                    'condition': action['condition'],
                    'ifBody': action['ifBody'],
                    'elseBody': el}]
            new_bodies.append(new_body[:index] +
                              new +
                              new_body[index:])
    return new_bodies


def get_list_of_delta_bodies(body: list, keep_empty_body=False, unwrap: bool = False):
    """
    :param body: list of actions
    :param keep_empty_body: if True, then empty bodies will also be present in the
    final list
    :param unwrap: if True, then unwrapped bodies of conditionals and loops will also
    be present in the final list
    :return: list of delta bodies
    """
    delta_bodies = []
    for index, action in enumerate(body):
        if action['type'] in actions:
            new_body = remove_basic_action(body, action, index)
            delta_bodies.append(new_body)
        elif action['type'] in simple_control:
            new_bodies = simple_inside_remove(body, action, index, keep_empty_body)
            delta_bodies += new_bodies
            if unwrap:
                new_bodies = do_unwrap(body, action, index)
                delta_bodies += new_bodies
        elif action['type'] == 'ifElse':
            new_bodies = complex_inside_remove(body, action, index, keep_empty_body)
            delta_bodies += new_bodies
            if unwrap:
                new_bodies = do_unwrap(body, action, index)
                delta_bodies += new_bodies
        else:
            raise ValueError('Unknown action type: {}'.format(action['type']))

    return delta_bodies


def get_one_level_delta_debugging_codes(code: Code, keep_empty_body=False,
                                        unwrap: bool = False):
    """

    :param code: Code object
    :param keep_empty_body: if True, then empty bodies will also be present in the
    final list
    :param unwrap: if True, then unwrapped bodies of conditionals and loops will also
    be present in the final list
    :return: list of mutated codes
    """
    body = code.astJson['body']
    delta_bodies = get_list_of_delta_bodies(body, keep_empty_body, unwrap)
    delta_codes = [{'type': 'run',
                    'body': body} for body in delta_bodies
                   if check_body_redundancy(body)]
    delta_codes = [json.loads(x) for x in
                   set([json.dumps(code, sort_keys=True) for code in
                        delta_codes])]
    return delta_codes


def check_codetask_redundancy_and_delta(code: Code, task: Task, keep_empty_body=False,
                                        unwrap: bool = False,
                                        delta_only: bool = False) -> bool:
    """
    :param code: Code object
    :param task: Task object
    :param keep_empty_body: if True, then empty bodies will also be present in the
    final list
    :param unwrap: if True, then unwrapped bodies of conditionals and loops will also
    be present in the final list
    :param delta_only: if True, then only the delta deubgging test will be run
    :return: True if the code is redundant, False otherwise
    """
    if not delta_only and not check_code_redundancy(code):
        return True
    delta_codes = get_one_level_delta_debugging_codes(code, keep_empty_body, unwrap)
    executor = Executor()
    for delta_code in delta_codes:
        res = executor.execute(task,
                               Code.parse_json({'program_type': 'karel',
                                                'program_json': delta_code}))
        if res['task_success']:
            #print('Delta code: {}'.format(delta_code))
            return True
    return False

# if __name__ == "__main__":
#     code = {
#         'type': 'run',
#         "body": [
#             {"type": "repeat", "times": 8,
#              "body": [
#                  {"type": "ifElse",
#                   "condition": {"type": "noMarkersPresent"},
#                   "ifBody": [{'type': 'move'},
#                              {'type': 'move'},
#                              {"type": "putMarker"}, ],
#                   "elseBody": [{"type": "pickMarker"}]},
#              ]}
#         ]}
#
#     delta_codes = get_one_level_delta_debugging_codes(Code({'program_type': 'karel',
#                                                             'program_json': code_json1}),
#                                                       keep_empty_body=False,
#                                                       unwrap=True)
#
#     for code in delta_codes:
#         print(code)

# print(check_code_redundancy(Code(code_json1)))
