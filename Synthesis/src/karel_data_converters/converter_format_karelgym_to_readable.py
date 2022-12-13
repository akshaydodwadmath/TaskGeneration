import json

import numpy as np

from src.datagen.utils.codegraph import ASTNode, ast_to_json

conditional_converter = {
    'frontIsClear': 'bool_path_ahead',
    'leftIsClear': 'bool_path_left',
    'rightIsClear': 'bool_path_right',
    'markersPresent': 'bool_marker_present',
    'noMarkersPresent': 'bool_no_marker_present',
    'notFrontIsClear': 'bool_no_path_ahead',
    'notLeftIsClear': 'bool_no_path_left',
    'notRightIsClear': 'bool_no_path_right'
}

others_converter = {
    'move': 'move',
    'turnLeft': 'turn_left',
    'turnRight': 'turn_right',
    'pickMarker': 'pick_marker',
    'putMarker': 'put_marker',
    'while': 'while',
    'repeat': 'repeat',
    'if': 'if',
    'ifElse': 'ifelse',
    'run': 'run',
}


def get_task_string(karelgym_json):
    start_string = 'type\tkarel\ngridsz\t(ncol=18,nrow=18)\nnumber_of_grids\t'
    num_examples = len(karelgym_json["examples"])
    start_string += str(num_examples)
    start_string += '\n'

    maxnumblocks = karelgym_json["num_blocks_allowed"]
    maxnumblocks_str = "maxnumblocks\t" + str(maxnumblocks) + '\n'
    blocksallowed_str = "blocksallowed\t" + karelgym_json[
        "type_blocks_allowed"] + '\n\n'  # additional new-line because after this, we have the first task grid
    start_string = start_string + maxnumblocks_str + blocksallowed_str

    task_grid_list = karelgym_json["examples"]
    for id, ele in enumerate(task_grid_list):

        taskstr = ''
        pregrid_str = ''
        postgrid_str = ''

        pregrid = ele["inpgrid_json"]
        pregrid_agent = pregrid["hero"].split(":")
        postgrid = ele["outgrid_json"]
        postgrid_agent = postgrid["hero"].split(":")

        pregrid_mat = np.full([int(pregrid["rows"]), int(pregrid["cols"])], '.',
                              dtype=str)
        pregrid_walls = pregrid["blocked"].split(' ')
        pregrid_markers = pregrid["markers"].split(' ')
        if pregrid_walls == ['']:  # no walls
            pregrid_walls = []
        for w in pregrid_walls:
            coords = w.split(":")
            pregrid_mat[int(coords[0]), int(coords[1])] = '#'
        if pregrid_markers == ['']:  # no markers
            pregrid_markers = []
        for m in pregrid_markers:
            coords = m.split(":")
            pregrid_mat[int(coords[0]), int(coords[1])] = coords[2]

        postgrid_mat = np.full([int(postgrid["rows"]), int(postgrid["cols"])], '.',
                               dtype=str)
        postgrid_walls = postgrid["blocked"].split(' ')
        postgrid_markers = postgrid["markers"].split(' ')
        if postgrid_walls == ['']:  # no walls
            postgrid_walls = []
        for w in postgrid_walls:
            coords = w.split(":")
            postgrid_mat[int(coords[0]), int(coords[1])] = '#'
        if postgrid_markers == ['']:  # no markers
            postgrid_markers = []
        for m in postgrid_markers:
            coords = m.split(":")
            postgrid_mat[int(coords[0]), int(coords[1])] = coords[2]

        # populate the task string
        pregrid_coords = [str(i) for i in range(1, int(pregrid["cols"]) + 3)]
        pregrid_coords.insert(0, 'pregrid_' + str(id + 1))
        pregrid_init = "\t".join(pregrid_coords)
        wall_row = ['#' for i in range(1, pregrid["cols"] + 3)]
        pregrid_str = pregrid_str + pregrid_init + '\n'
        pregrid_str += '1\t' + '\t'.join(wall_row) + '\n'  # initial row of walls
        for r in range(int(pregrid["rows"])):
            r_str = str(r + 2) + '\t' + '#\t'  # for the border in the beginning
            r_ele = list(pregrid_mat[r, :])
            r_str = r_str + '\t'.join(r_ele) + '\t#\n'  # # is for the border in the end
            pregrid_str += r_str
        pregrid_str += str(int(pregrid["rows"]) + 2) + '\t' + '\t'.join(
            wall_row) + '\n'  # final row of walls
        pregrid_str += 'agentloc_' + str(id + 1) + '\t' + "(col=" + pregrid_agent[
            1] + ",row=" + pregrid_agent[0] + ")\n"
        pregrid_str += 'agentdir_' + str(id + 1) + '\t' + pregrid_agent[
            2] + "\n\n"  # additional newline before starting the postgrid

        # populate the task string
        postgrid_coords = [str(i) for i in range(1, int(postgrid["cols"]) + 3)]
        postgrid_coords.insert(0, 'postgrid_' + str(id + 1))
        postgrid_init = "\t".join(postgrid_coords)
        wall_row = ['#' for i in range(1, postgrid["cols"] + 3)]
        postgrid_str = postgrid_str + postgrid_init + '\n'
        postgrid_str += '1\t' + '\t'.join(wall_row) + '\n'  # initial row of walls
        for r in range(int(postgrid["rows"])):
            r_str = str(r + 2) + '\t' + '#\t'  # for the border in the beginning
            r_ele = list(postgrid_mat[r, :])
            r_str = r_str + '\t'.join(r_ele) + '\t#\n'
            postgrid_str += r_str

        postgrid_str += str(int(postgrid["rows"]) + 2) + '\t' + '\t'.join(
            wall_row) + '\n'  # final row of walls
        postgrid_str += 'agentloc_' + str(id + 1) + '\t' + "(col=" + postgrid_agent[
            1] + ",row=" + postgrid_agent[0] + ")\n"
        postgrid_str += 'agentdir_' + str(id + 1) + '\t' + postgrid_agent[
            2] + "\n\n"  # additional newline before starting the next pregrid

        taskstr = taskstr + pregrid_str + postgrid_str
        start_string = start_string + taskstr

    return start_string


def karelcode_json_to_ast(root):
    """ Converts a JSON dictionary to an ASTNode."""

    def get_children(json_node, if_flag=False, else_flag=False):
        if if_flag:
            children = json_node.get("ifBody", [])
        elif else_flag:
            children = json_node.get("elseBody", [])
        else:
            children = json_node.get("body", [])
        return children

    node_type = root["type"]

    if node_type == "run":
        run_children = root["body"]
        return ASTNode('run', None,
                           [karelcode_json_to_ast(child) for child in run_children])

    elif node_type == "ifElse":
        type = others_converter[node_type]
        condition = root["condition"]
        val = conditional_converter[condition]
        do_children = [karelcode_json_to_ast(c) for c in root["ifBody"]]
        else_children = [karelcode_json_to_ast(c) for c in root["elseBody"]]
        return ASTNode(type, val, [ASTNode('do', val, do_children),
                                   ASTNode('else', val, else_children)])
    elif node_type == "if":
        type = others_converter[node_type]
        condition = root["condition"]
        val = conditional_converter[condition]
        do_children = [karelcode_json_to_ast(c) for c in root["body"]]
        return ASTNode(type, val, [ASTNode('do', val, do_children)])

    # TODO add repeat until goal

    elif node_type == "repeat":
        type = others_converter[node_type]
        val = root["times"]
        children = [karelcode_json_to_ast(c) for c in root["body"]]
        return ASTNode(type, str(val), children)

    elif node_type == "while":
        type = others_converter[node_type]
        condition = root["condition"]
        val = conditional_converter[condition]
        children = [karelcode_json_to_ast(c) for c in root["body"]]
        return ASTNode(type, val, children)

    elif node_type == 'move':
        return ASTNode(others_converter[node_type])

    elif node_type == 'turnLeft':
        return ASTNode(others_converter[node_type])

    elif node_type == 'turnRight':
        return ASTNode(others_converter[node_type])

    elif node_type == 'pickMarker':
        return ASTNode(others_converter[node_type])

    elif node_type == 'putMarker':
        return ASTNode(others_converter[node_type])
    else:
        assert "Unknown node type encountered!"

    return None


def get_code_json(karelgym_json):
    code_kareljson = karelgym_json['program_json']
    ast_code = karelcode_json_to_ast(code_kareljson)
    code_json = ast_to_json(ast_code)

    return code_json


# Generate code-task pairs for qualitative evaluation
if __name__ == "__main__":
    from tqdm import tqdm
    import random

    test_task_json = "../../datasets/synthetic/karelgym_10k/train.json"
    task_id = 0
    inner_i = 0
    nb_qual_tasks = 50
    nums = random.sample(range(0, 10000), nb_qual_tasks)
    print(nums)
    with open(test_task_json, 'r') as fp:
        for example in tqdm(fp):
            taskjson = json.loads(example)
            task_id += 1
            if task_id in nums:
                # print("Task string:")
                # print(get_task_string(taskjson))
                f = open(f"codetasks/{inner_i}_task.txt", "w")
                f.write(get_task_string(taskjson))
                # print("Code json:")
                # print(get_code_json(taskjson))
                f = open(f"codetasks/{inner_i}_code.json", "w")
                # print(type(get_code_json(taskjson)))
                f.write(
                    json.dumps(get_code_json(taskjson), ensure_ascii=False, indent=4))
                inner_i += 1
