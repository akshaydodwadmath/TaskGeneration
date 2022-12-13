import ast
import json
import time
from typing import Tuple

import networkx as nx
import numpy as np

from src.karel_codetask_scoring.execquality import compute_codetask_quality_from_actions
from src.karel_emulator.code import Code
from src.karel_emulator.task import Task
from src.karel_emulator.tokens import actions
from src.karel_emulator.world import World

dir_to_num = {
    'east': 1,
    'south': 4,  # 2
    'west': 3,
    'north': 2,  # 4

    'any': -1
}

dir_dict = {
    '11': [],
    '13': ['turnRight', 'turnRight'],
    '12': ['turnLeft'],  # 14
    '14': ['turnRight'],  # 12

    '33': [],
    '31': ['turnLeft', 'turnLeft'],
    '32': ['turnRight'],  # 34
    '34': ['turnLeft'],  # 32

    '44': [],
    '42': ['turnRight', 'turnRight'],
    '43': ['turnRight'],  # 41
    '41': ['turnLeft'],  # 43

    '22': [],
    '24': ['turnLeft', 'turnLeft'],
    '23': ['turnLeft'],  # 21
    '21': ['turnRight']  # 23
}


def get_neighbors(x_coord, y_coord, dir, grid, gridsize_row, gridsize_col):
    neighbors = []
    dx = 0
    dy = 0
    if grid[x_coord, y_coord] == 1:
        return neighbors
    if dir == 1:  # east
        dx = 0
        dy = 1
    if dir == 2:  # north
        dx = 1
        dy = 0
    if dir == 3:  # west
        dx = 0
        dy = -1
    if dir == 4:  # south
        dx = -1
        dy = 0
    else:
        assert "Invalid dir encountered"

    for i in range(1, 5):  # add all the directions in the same loc
        if i == dir:
            continue
        neighbors.append([x_coord, y_coord, i])
    # add the neighbor for the move option
    new_x_coord = x_coord + dx
    new_y_coord = y_coord + dy

    if new_x_coord > gridsize_row - 1 or new_y_coord > gridsize_col - 1:
        return neighbors
    elif new_x_coord < 0 or new_y_coord < 0:
        return neighbors

    elif grid[new_x_coord, new_y_coord] == 1:
        return neighbors

    else:
        neighbors.append([new_x_coord, new_y_coord, dir])
        return neighbors


def construct_graph_from_grid(grid):
    node_dict = {}

    k = 0
    # add all the elements in the node dictionary
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if [i, j, 1] not in node_dict.values():
                node_dict[k] = str([i, j, 1])
                k += 1
            if [i, j, 2] not in node_dict.values():
                node_dict[k] = str([i, j, 2])
                k += 1
            if [i, j, 3] not in node_dict.values():
                node_dict[k] = str([i, j, 3])
                k += 1
            if [i, j, 4] not in node_dict.values():
                node_dict[k] = str([i, j, 4])
                k += 1

    inv_node_map = {v: k for k, v in node_dict.items()}

    # generate the edge list
    edge_list = []
    for k, item in node_dict.items():
        tuple = eval(item)
        tuple = [int(ele) for ele in tuple]

        nbs = get_neighbors(tuple[0], tuple[1], tuple[2], grid, grid.shape[0],
                            grid.shape[1])
        nbs = [str(ele) for ele in nbs]

        edge_node_1 = inv_node_map[item]
        for ele in nbs:
            edge_node_2 = inv_node_map[ele]
            edge_list.append((edge_node_1, edge_node_2))

    # construct the graph from edge list
    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    return G, inv_node_map, node_dict, edge_list


def get_shortest_path(g, start, goal, node_dict, inv_node_map):
    all_shortest_paths = []
    start_node = inv_node_map[str(start)]
    for i in range(1, 5):  # get paths to all the orientations in the goal state
        goal_o = str([goal[0], goal[1], i])
        goal_node = inv_node_map[str(goal_o)]
        shortest_path = nx.shortest_path(g, source=start_node, target=goal_node)

        shortest_blocks = [node_dict[ele] for ele in shortest_path]
        all_shortest_paths.append([i, shortest_blocks])

    return all_shortest_paths


def get_turns_to_next_loc(init_loc, next_loc):
    action = []
    if init_loc[0] == next_loc[0] and init_loc[1] == next_loc[1]:
        dir_pair = str(init_loc[2]) + str(next_loc[2])
        action.extend(dir_dict[dir_pair])
    else:
        action.append('move')

    return action


def prune_path_to_remove_walking_over_walls(path):
    new_path = []
    for ele in path:
        nele = ast.literal_eval(ele)
        if int(nele[0]) % 2 == 0:
            continue
        elif int(nele[1]) % 2 == 0:
            continue
        else:
            new_path.append(ele)

    return new_path


def get_blocks_from_path(path):
    blocks = []
    # prune the path to avoid additional moves
    # pruned_path  = prune_path_to_remove_walking_over_walls(path)
    pruned_path = path
    for i in range(len(pruned_path) - 1):
        state = eval(pruned_path[i])
        state = [int(ele) for ele in state]
        next_state = eval(pruned_path[i + 1])
        next_state = [int(ele) for ele in next_state]

        blocks.extend(get_turns_to_next_loc(state, next_state))

    return blocks


def get_shortest_block_seq_to_subgoal(g, vertice_dict, inv_vertice_map, edge_list,
                                      start_state, end_loc):
    all_shortest_paths = {}
    for ele in end_loc:
        all_shortest_paths[str(ele)] = get_shortest_path(g, start_state, ele,
                                                         vertice_dict, inv_vertice_map)

    min_count = np.inf
    min_block_seq = []
    min_path = []
    dir = None
    for ele in end_loc:
        for path in all_shortest_paths[str(ele)]:
            blocks = get_blocks_from_path(path[1])
            if len(blocks) < min_count:
                min_count = len(blocks)
                min_block_seq = blocks
                min_path = path
                dir = path[0]

    return min_count, min_block_seq, min_path, dir


def get_karel_shortest_path_length(pregrid, postgrid, verbose=0):
    gridsz, start_loc, start_dir, \
    pregrid_blocked, end_loc, end_dir, \
    postgrid_blocked, subgoals, diff = \
        get_grid_mat(pregrid, postgrid)

    if verbose == 1:
        print("Gridsize:", gridsz)
        print("Start:", start_loc, start_dir)
        print("End:", end_loc, end_dir)
        print("pregrid:", pregrid_blocked)
        print("postgrid:", postgrid_blocked)
        print("diff:", diff)
        print("subgoals:", subgoals)

    g, inv_vertice_map, vertice_dict, edge_list = construct_graph_from_grid(
        pregrid_blocked)

    completed_goals = []
    start_state = [start_loc[0], start_loc[1], dir_to_num[start_dir]]
    start_xy = start_loc
    total_steps = 0
    full_path = []
    full_block_seq = []
    for i in range(len(subgoals)):
        goals_remaining = list(set(subgoals) - set(completed_goals))
        # get the subgoal with minimum distance
        dists = []
        for ele in goals_remaining:
            dists.append(abs(start_xy[0] - ele[0]) + abs(start_xy[1] - ele[1]))
        goal = goals_remaining[np.argmin(dists)]

        if verbose == 1:
            print("Next goal:", goal)

        completed_goals.append(goal)
        min_count, min_block_seq, \
        min_path, dir = \
            get_shortest_block_seq_to_subgoal(g,
                                              vertice_dict,
                                              inv_vertice_map,
                                              edge_list,
                                              start_state,
                                              [goal])

        if verbose == 1:
            print("Shortest Path:", min_block_seq)

        total_steps = total_steps + min_count + 1  # +1 for marker activity
        if diff[goal[0]][goal[1]] > 0:
            full_path.extend(
                min_path[1] + ['putMarker' for _ in range(diff[goal[0]][goal[1]])])
            full_block_seq.extend(
                min_block_seq + ['putMarker' for _ in range(diff[goal[0]][goal[1]])])
        else:
            full_path.extend(min_path[1] + ['pickMarker' for _ in range(-diff[goal[
                0]][goal[1]])])
            full_block_seq.extend(min_block_seq + ['pickMarker' for _ in range(-diff[
                goal[0]][goal[1]])])

        start_state = [goal[0], goal[1], dir]
        start_xy = goal

    ####### Add the final path to the goal state
    if verbose == 1:
        print("End state:", end_loc, end_dir)
    min_count, min_block_seq, \
    min_path, dir = \
        get_shortest_block_seq_to_subgoal(g, vertice_dict, inv_vertice_map,
                                          edge_list, start_state, [end_loc])
    if verbose == 1:
        print("Shortest Path:", min_block_seq)

    # TODO: this needs to be modified for hoc vs karel
    if dir != dir_to_num[end_dir] and dir_to_num[end_dir] != -1:
        key = str(dir) + str(dir_to_num[end_dir])
        add_actions = dir_dict[key]

        total_steps = total_steps + min_count + len(add_actions)
        full_path.extend(
            min_path[1] + ['turn' + str(k) for k in range(len(add_actions))])

        full_block_seq.extend(min_block_seq)
        full_block_seq.extend(add_actions)
    else:
        total_steps = total_steps + min_count
        full_path.extend(min_path[1])

        full_block_seq.extend(min_block_seq)

    if verbose == 1:
        print("Total steps:", total_steps)
        print("Full Path:", full_path)
        print("Full block sequence:", full_block_seq)

    return total_steps, full_block_seq


def get_grid_mat(pregrid: World, postgrid: World):
    gridsize_start = (pregrid.rows, pregrid.cols)
    karelstart = (pregrid.heroRow, pregrid.heroCol)
    karelstart_dir = pregrid.heroDir

    gridsize_end = (postgrid.rows, postgrid.cols)
    karelend = (postgrid.heroRow, postgrid.heroCol)
    karelend_dir = postgrid.heroDir

    if gridsize_start != gridsize_end:
        assert "Check the dimensions of the start and end grids!"
        return -1

    mat_pregrid = pregrid.blocked
    mat_postgrid = postgrid.blocked

    mat_pregrid_markers = pregrid.markers
    mat_postgrid_markers = postgrid.markers

    # Find the difference of the 2 mats: postgrid-pregrid
    subgoals = []
    mat_diff = np.where(mat_pregrid_markers != mat_postgrid_markers)
    list_of_coordinates = list(zip(mat_diff[0], mat_diff[1]))
    diff = np.zeros((gridsize_start[0], gridsize_start[1]), dtype=np.int64)
    for coord in list_of_coordinates:
        subgoals.append(coord)
        diff[coord[0], coord[1]] = mat_postgrid_markers[coord[0], coord[1]] - \
                                   mat_pregrid_markers[coord[0], coord[1]]

    return gridsize_start, karelstart, karelstart_dir, \
           mat_pregrid, karelend, karelend_dir, \
           mat_postgrid, subgoals, diff


def check_marker_diffs(pregrid: World, postgrid: World):
    return (pregrid.markers != postgrid.markers).any()


def check_shortest_path(code: Code,
                        task: Task,
                        ignore_marker_grids: bool = True,
                        verbose=0,
                        check_path_quality: bool = False) -> Tuple[bool, list]:
    """
    :param code: Code object
    :param task: Task object
    :param ignore_marker_grids: If True, then the shortest path is checked only for
    the grids without marker differences
    :param verbose: If 1, then prints the shortest path
    :param check_path_quality: If True, then checks the quality of the shortest path
    :return: True, if NO shorter seq of basic actions can solve the tasks; False otherwise
    """

    # obtain the status for each taskgrid in the json file
    qualities = []
    for pregrid, postgrid in zip(task.pregrids, task.postgrids):
        if ignore_marker_grids and check_marker_diffs(pregrid, postgrid):
            return True, [1.0]
        length, full_path = get_karel_shortest_path_length(pregrid, postgrid,
                                                           verbose=verbose)
        if length < code.total_count:
            return False, [0.0]
        basic_actions_only = all([action['type'] for action in code.astJson if action in
                                  actions])
        if not basic_actions_only and length == code.total_count:
            return False, [0.0]
        if check_path_quality:
            qualities.append(compute_codetask_quality_from_actions(full_path,
                                                                   max(pregrid.rows,
                                                                       pregrid.cols),
                                                                   task.type)[0])
        else:
            qualities.append(1.0)
    return True, qualities


# TODO: check if code is optimized and cases where markers are present are avoided
#  check for time too

if __name__ == "__main__":
    test = '{"task_type": "hoc", "examples": [{"example_index": 0, "inpgrid_json": {' \
           '"blocked": "4:5 3:8 1:6 1:8 0:5", "cols": 11, "crashed": false, "hero": ' \
           '"1:4:west", "markers": "3:4:1 3:5:2 3:6:4 3:7:5 2:0:2 2:8:9 1:3:1 1:5:1 ' \
           '1:9:1 0:0:4 0:3:5 0:7:3", "rows": 5}, "outgrid_json": {"blocked": "4:5 ' \
           '3:8 1:6 1:8 0:5", "cols": 11, "crashed": false, "hero": "1:3:any", ' \
           '"markers": "3:4:1 3:5:2 3:6:4 3:7:5 2:0:2 2:8:9 1:4:1 1:5:1 1:9:1 0:0:4 ' \
           '0:3:4 0:7:3", "rows": 5}}], "solution": {"program_type":"hoc", ' \
           '"program_json":{"body": [{"type": "putMarker"}, {"type": "move"}, ' \
           '{"type": ' \
           '"pickMarker"}, {"body": [{"type": "turnLeft"}, {"type": "move"}, {"type": ' \
           '"turnLeft"}], "condition": {"type": "noMarkersPresent"}, "type": ' \
           '"while"}, {"type": "turnLeft"}, {"body": [{"body": [{"type": ' \
           '"pickMarker"}], "condition": {"condition": {"type": "rightIsClear"}, ' \
           '"type": "not"}, "type": "if"}, {"type": "turnRight"}], "times": 5, ' \
           '"type": "repeat"}, {"type": "turnLeft"}, {"type": "move"}]}}, ' \
           '"num_examples": 1, "num_blocks_allowed": 1000, "type_blocks_allowed": ' \
           '"if,ifelse,while,repeat,putMarker,pickMarker"} '

    # test_task_file = '../../tests/datasets/code_task.json'
    # with open(test_task_file, 'r') as fp:
    karelobj_json = json.loads(test)

    task = Task.parse_json(karelobj_json)
    print(task.pregrids[0].toString())
    print()
    print(task.postgrids[0].toString())

    code = Code.parse_json(karelobj_json['solution'])

    start = time.time()
    print(check_shortest_path(code, task, verbose=1,
                              ignore_marker_grids=False,
                              check_path_quality=True))
    print(time.time() - start)
