import json
import re

import numpy as np




def readable_codejson_to_karelgym_codejson(readable_code: json):
    ''' Converts a ICLR18 JSON dictionary to an ASTNode.'''

    # dictionary to convert tokens
    bool_converter = {
        'bool_marker_present': 'markersPresent',
        'bool_no_marker_present': 'noMarkersPresent',
        'bool_path_ahead': 'frontIsClear',
        'bool_no_path_ahead': 'notFrontIsClear',
        'bool_path_left': 'leftIsClear',
        'bool_no_path_left': 'notLeftIsClear',
        'bool_path_right': 'rightIsClear',
        'bool_no_path_right': 'notRightIsClear',
        'bool_goal': 'boolGoal',
        '1': "1",
        '2': "2",
        '3': "3",
        '4': "4",
        '5': "5",
        '6': "6",
        '7': "7",
        '8': "8",
        '9': "9",
        '10': "10",
    }

    action_converter = {
        'move': 'move',
        'turn_right': 'turnRight',
        'turn_left': 'turnLeft',
        'put_marker': 'putMarker',
        'pick_marker': 'pickMarker'
    }

    def get_children(json_node):
        children = json_node.get('children', [])
        return children

    node_type = readable_code['type']
    children = get_children(readable_code)

    if node_type == "run":
        children_list = [readable_codejson_to_karelgym_codejson(c) for c in children]
        return {"type":'run', "body":children_list}

    if "(" in node_type:
        node_type = node_type.split('(')[0]
        node_cond = node_type.split('(')[1][:-1]
        cond = bool_converter[node_cond]
        if node_type == 'ifelse':
            assert (len(children) == 2)  # Must have do, else nodes for children


            do_node = children[0]
            assert (do_node['type'] == 'do')
            else_node = children[1]
            assert (else_node['type'] == 'else')
            do_list = [readable_codejson_to_karelgym_codejson(child) for child in get_children(do_node)]
            else_list = [readable_codejson_to_karelgym_codejson(child) for child in get_children(else_node)]
            # node_type is 'maze_ifElse_isPathForward' or 'maze_ifElse_isPathLeft' or 'maze_ifElse_isPathRight'
            return { "type":"ifelse", "condition": cond,
                "body": [{"type":'do', "condition":cond, "body": do_list},
                         {"type":'else', "condition":cond, "body":else_list}]
            }

        elif node_type == 'if':
            assert (len(children) == 1)  # Must have condition, do nodes for children
            do_node = children[0]
            assert (do_node['type'] == 'do')

            do_list = [readable_codejson_to_karelgym_codejson(child) for child in get_children(do_node)]

            return {
                "type":'if',
                "condition": cond,
                "body": do_list
            }



        elif node_type == 'while':

            while_list = [readable_codejson_to_karelgym_codejson(child) for child in children]
            return {
                "type": 'while',
                "condition": cond,
                "body": while_list
            }

        elif node_type == 'repeat':
            repeat_list = [readable_codejson_to_karelgym_codejson(child) for child in children]
            return {
               "type":'repeat',
               "times": cond,
               "body": repeat_list,
            }

        elif node_type == 'repeat_until_goal':

            repeat_until_goal_list = [readable_codejson_to_karelgym_codejson(child) for child in children]
            return {
                "type":'repeat_until_goal',
                "condition": 'bool_goal',
                "body": repeat_until_goal_list
            }

    if node_type == 'move':
        return {"type":action_converter[node_type]}

    if node_type == 'turn_left':
        return {"type":action_converter[node_type]}

    if node_type == 'turn_right':
        return {"type":action_converter[node_type]}

    if node_type == 'pick_marker':
        return {"type":action_converter[node_type]}

    if node_type == 'put_marker':
        return {"type":action_converter[node_type]}

    print('Unexpected node type, failing:', node_type)
    assert (False)


## get all the task elements in a dictionary: grids, maxnumblocks, block_types, ngrids
def get_task_elements(taskfile: str):
    with open(taskfile, 'r') as fp:

        # obtain the type of the task
        type_field = fp.readline().rstrip()
        if re.match(r'type\tkarel', type_field):
            type = 'karel'
        elif re.match(r'type\thoc', type_field):
            type = 'hoc'
        else:
            assert (False)

        # obtain the overall gridsz
        gridsz_field = fp.readline()
        gridsz = re.match(r'gridsz\t\(ncol=(?P<c>\d+),nrow=(?P<r>\d+)\)', gridsz_field)
        r = int(gridsz.group('r'))
        c = int(gridsz.group('c'))

        # obtain the number of grids
        ngrids_field = fp.readline()
        ngrids_re = re.match(r'number_of_grids\t(?P<ngrids>\d+)', ngrids_field)
        ngrids = int(ngrids_re.group('ngrids'))

        # obtain the maxnumblocks
        maxnumblocks_field = fp.readline()
        maxnumblocks_re = re.match(r'maxnumblocks\t(?P<maxnumblocks>\d+)',
                                   maxnumblocks_field)
        maxnumblocks = int(maxnumblocks_re.group('maxnumblocks'))

        # obtain the blocks allowed
        blocksallowed_field = fp.readline()
        blocksallowed_field = blocksallowed_field.strip('\n')
        blocks_allowed_str_old = blocksallowed_field.split('\t')[1]
        blocks_allowed = blocks_allowed_str_old.split(',')
        blocks_allowed[:] = [x if x != 'repeat_until_goal' else 'while' for x in
                             blocks_allowed]
        blocks_allowed[:] = [x if x != 'hoc_actions' else 'move,turnLeft,turnRight' for
                             x in blocks_allowed]
        blocks_allowed[:] = [
            x if x != 'karel_actions' else 'move,turnLeft,turnRight,pickMarker,putMarker'
            for x in blocks_allowed]
        blocks_allowed_str = ','.join(blocks_allowed)

        def encode(cell):
            if cell == '#':
                return 0
            elif cell == '.':
                return 1
            elif cell == '+':
                return 4
            elif cell == 'x' or cell in ["1", "2", "3", "4", "5", "6",
                                         "7", "8", "9", "10", "11", "12",
                                         "13", "14", "15", "16", "17", "18", "19", "20"] :
                return 4
            elif cell in map(str, range(2, 10)):
                return int(cell) + 3
            else:
                assert (False)

        def read_grid():
            grid = []
            no_grid_line = False

            i = 0
            while not no_grid_line:
                prev_line = fp.tell()
                row = fp.readline()
                row = row[:-1]
                num = i + 1
                raw_row = re.match(f'{num}\t(?P<tail>.*)', row)
                if raw_row is None:
                    no_grid_line = True
                else:
                    row = raw_row.group('tail')
                    row = list(map(encode, row.split('\t')))
                    grid.append(row)
                i += 1
            fp.seek(prev_line)
            return grid

        # obtain the list of ngrids of the task
        grids = []  # [(ncol, nrow, numpy_pre, agent_pre, agentdir_pre, numpy_post, agent_post, agentdir_post)]
        for grid_id in range(1, ngrids + 1):
            # read the pregrid details
            fp.readline()
            fp.readline()

            pregrid = read_grid()  # convert it into a numpy array
            pregrid_numpy = np.array(pregrid)
            agentloc_pre_field = fp.readline()
            agentloc_pre = re.match(
                f'agentloc_{grid_id}\t\(col=(?P<c>\d+),row=(?P<r>\d+)\)',
                agentloc_pre_field)
            agent_r_pregrid = int(agentloc_pre.group('r')) - 1
            agent_c_pregrid = int(agentloc_pre.group('c')) - 1
            agentdir_pre_field = fp.readline()
            agentdir_pre = re.match(f'agentdir_{grid_id}\t(?P<d>.*)',
                                    agentdir_pre_field)
            agentdir_pregrid = agentdir_pre.group('d')

            # read the postgrid details
            fp.readline()
            fp.readline()

            postgrid = read_grid()
            postgrid_numpy = np.array(postgrid)
            agentloc_post_field = fp.readline()
            agentloc_post = re.match(
                f'agentloc_{grid_id}\t\(col=(?P<c>\d+),row=(?P<r>\d+)\)',
                agentloc_post_field)
            agent_r_postgrid = int(agentloc_post.group('r')) - 1
            agent_c_postgrid = int(agentloc_post.group('c')) - 1
            agentdir_post_field = fp.readline()
            agentdir_post = re.match(f'agentdir_{grid_id}\t(?P<d>.*)',
                                     agentdir_post_field)
            agentdir_postgrid = agentdir_post.group('d')

            # grids.append((pregrid_numpy.shape[0], pregrid_numpy.shape[1], pregrid_numpy, (agent_c_pregrid, agent_r_pregrid), agentdir_pregrid,
            #               postgrid_numpy, (agent_c_postgrid, agent_r_postgrid), agentdir_postgrid))
            pregrid_formatted = get_grid_elements(
                (pregrid_numpy, (agent_r_pregrid, agent_c_pregrid), agentdir_pregrid))
            postgrid_formatted = get_grid_elements((postgrid_numpy, (
            agent_r_postgrid, agent_c_postgrid), agentdir_postgrid))
            # grid_dict
            grid_dict = {}
            # grid_dict['example_index'] = grid_id - 1
            grid_dict['inpgrid_json'] = pregrid_formatted
            grid_dict['outgrid_json'] = postgrid_formatted
            grids.append(grid_dict)

    task_info = {}
    task_info['grids'] = grids
    task_info['maxnumblocks'] = maxnumblocks
    task_info['ngrids'] = ngrids
    task_info['blocks_allowed_str'] = blocks_allowed_str
    task_info['type'] = type
    return task_info


## get all the GRID elements in a dictionary: walls, markers, agent, rows, cols of A grid (visual puzzle)
def get_grid_elements(grid_details):
    '''

    :param grid_details: (taskgrid_numpy, agent_loc, agent_dir)
    :return:
    '''

    grid_dict = {}
    taskgrid = grid_details[0]
    agent_loc = grid_details[1]
    agent_dir = grid_details[2]
    hero = str(agent_loc[0]) + ':' + str(agent_loc[1]) + ':' + str(agent_dir)
    crashed = 'false'
    rows = taskgrid.shape[0]
    cols = taskgrid.shape[1]

    # get the index of walls in the taskgrid
    walls_ids = np.where(taskgrid == 0)
    walls_coords = list(zip(walls_ids[0], walls_ids[1]))
    walls_str = ''
    for i, coord in enumerate(walls_coords):
        walls_str += str(coord[0])
        walls_str += ":"
        walls_str += str(coord[1])
        if i != len(walls_coords) - 1:
            walls_str += " "

    # get the index of markers in the task grid
    marker_ids = np.where(taskgrid >= 4)
    marker_coords = list(zip(marker_ids[0], marker_ids[1]))
    marker_str = ''
    for i, coord in enumerate(marker_coords):
        marker_str += str(coord[0])
        marker_str += ":"
        marker_str += str(coord[1])
        marker_str += ":"
        if taskgrid[coord[0], coord[1]] == 4:  # single marker in the location
            marker_str += str(1)
        else:
            marker_str += str(
                taskgrid[coord[0], coord[1]])  # more than one marker in the location
        if i != len(marker_coords) - 1:
            marker_str += " "

    # populate the grid_dict
    grid_dict['blocked'] = walls_str
    grid_dict['cols'] = int(cols)
    grid_dict['crashed'] = crashed
    grid_dict['hero'] = hero
    grid_dict['markers'] = marker_str
    grid_dict['rows'] = int(rows)

    return grid_dict


# get the JSON obj from a (task.txt, code.json) file pair
def get_benchmark_format(taskfile, codefile):
    task_elements = get_task_elements(taskfile)

    with open(codefile, 'r') as fp:
        codeast_json = json.load(fp)
        # codeast = json_to_ast(codeast_json)
        # code_benchmark_format = ast_to_json_karelgym_format(codeast)

    benchmark_task_json = {}
    benchmark_task_json['type'] = task_elements['type']
    benchmark_task_json['examples'] = task_elements['grids']
    benchmark_task_json['solution'] = {'program_type': task_elements['type'],
                                           'program_json': readable_codejson_to_karelgym_codejson(codeast_json)}
    benchmark_task_json['num_examples'] = task_elements['ngrids']
    benchmark_task_json['num_blocks_allowed'] = task_elements['maxnumblocks']
    benchmark_task_json['type_blocks_allowed'] = task_elements['blocks_allowed_str']

    return benchmark_task_json


if __name__ == "__main__":
    IN_FOLDER_HOC = '../../tests/datasets/'
    IN_FOLDER_KAREL = IN_FOLDER_HOC
    OUT_FOLDER_HOC = IN_FOLDER_HOC
    OUT_FOLDER_KAREL = IN_FOLDER_HOC
    task_list_hoc = ['in-hoc-I']
    task_list_karel = ['in-karel-E']

    # convert all the HOC tasks into 1 file
    with open(OUT_FOLDER_HOC + 'train_hoc.json', 'w') as fp:
        for ele in task_list_hoc:
            print("Converting:", ele)
            taskfile = IN_FOLDER_HOC + ele + '_task.txt'
            codefile = IN_FOLDER_HOC + ele + '_code.json'
            benchmark_format = get_benchmark_format(taskfile, codefile)
            benchmark_format_str = json.dumps(benchmark_format)
            fp.write(benchmark_format_str + '\n')

    # convert all the Karel tasks into 1 file
    with open(OUT_FOLDER_KAREL + 'train_karel.json', 'w') as fp:
        for ele in task_list_karel:
            print("Converting:", ele)
            taskfile = IN_FOLDER_KAREL + ele + '_task.txt'
            codefile = IN_FOLDER_KAREL + ele + '_code.json'
            benchmark_format = get_benchmark_format(taskfile, codefile)
            benchmark_format_str = json.dumps(benchmark_format)
            fp.write(benchmark_format_str + '\n')
