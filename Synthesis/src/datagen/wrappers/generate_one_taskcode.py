import random
from numpy import random as nprand
from src.datagen.codegen.generate_code import generate_one_code
from src.datagen.taskgen.generate_task import generate_task_grids
from src.datagen.utils.codegraph import json_to_ast, ast_to_json_karelgym_format
from src.datagen.wrappers.config_one_taskcode import DIST_NUMGRIDS, DIST_MARKERCOUNTS, DIST_PROGRAM_TYPE, \
    DIST_PWALL, DIST_PMARKER, GLOBAL_WHILECOUNTER_MIN, \
    GLOBAL_WHILECOUNTER_MAX, GLOBAL_MINROWS, GLOBAL_MAXROWS, GLOBAL_MINCOLS, \
    GLOBAL_MAXCOLS, \
    MAXITERS_TASKGRID, STORE
from src.karel_codetask_scoring.deltadebugging import check_code_redundancy, check_codetask_redundancy_and_delta
from src.karel_codetask_scoring.shortestpath import check_shortest_path
from src.karel_codetask_scoring.coverage import compute_coverage
from src.karel_codetask_scoring.solvability import check_solvability
from src.karel_emulator.code import Code
from src.karel_emulator.task import Task


def convert_task_grid_into_json(task_grids: list, task_type:str):
    # we have a list of Grid class objects that need to be converted into a list of dictionaries
    examples = []
    for ele in task_grids:
        # pregrid processing
        inpgrid_json = {}
        inpgrid_json['rows'] = ele._nrows
        inpgrid_json['hero'] = str(ele._initloc_avatar[0]) + ":" + str(
            ele._initloc_avatar[1]) + ":" + ele._initdir_avatar
        inpgrid_json['cols'] = ele._ncols
        inpgrid_json['markers'] = ''
        for key, val in ele._markercells['pregrid'].items():
            curr_str = str(key[0]) + ':' + str(key[1]) + ':' + str(val)
            inpgrid_json['markers'] += curr_str + ' '

        inpgrid_json['markers'] = inpgrid_json[
            'markers'].rstrip()  # removes the trailing space
        inpgrid_json['crashed'] = 'false'
        inpgrid_json['blocked'] = ''
        for w in ele._walls['pregrid']:
            curr_str = str(w[0]) + ':' + str(w[1])
            inpgrid_json['blocked'] += curr_str + ' '
        inpgrid_json['blocked'] = inpgrid_json['blocked'].rstrip()

        # postgrid processing
        outgrid_json = {}
        # Change direction of postgrid task object to any
        if task_type == "hoc":
            ele._currdir_avatar = "any"

        outgrid_json['rows'] = ele._nrows
        outgrid_json['hero'] = str(ele._currloc_avatar[0]) + ":" + str(
            ele._currloc_avatar[1]) + ":" + ele._currdir_avatar
        outgrid_json['cols'] = ele._ncols
        outgrid_json['markers'] = ''
        for key, val in ele._markercells['postgrid'].items():
            curr_str = str(key[0]) + ':' + str(key[1]) + ':' + str(val)
            outgrid_json['markers'] += curr_str + ' '

        outgrid_json['markers'] = outgrid_json[
            'markers'].rstrip()  # removes the trailing space
        outgrid_json['crashed'] = 'false'
        outgrid_json['blocked'] = ''
        for w in ele._walls['postgrid']:
            curr_str = str(w[0]) + ':' + str(w[1])
            outgrid_json['blocked'] += curr_str + ' '
        outgrid_json['blocked'] = outgrid_json['blocked'].rstrip()

        # add the task to examples
        examples.append({'inpgrid_json': inpgrid_json, 'outgrid_json': outgrid_json})

    return examples


def convert_to_karelgym_format_json(code_json: dict, task_grids: list, program_type: str):
    # convert the code into ASTNode
    code_ast = json_to_ast(code_json)
    # obtain the blocks_allowed, num_blocks info to be filled into the task json later
    marker_activity_flag = code_ast.marker_activity_flag()
    num_blocks = code_ast.size()
    blocks_allowed = []
    run_children = [c._type.split('(')[0] for c in code_ast.children()]
    if 'while' in run_children:
        blocks_allowed.append('while')
    if 'ifelse' in run_children:
        blocks_allowed.append('ifelse')
    if 'if' in run_children:
        blocks_allowed.append('if')
    if 'repeat' in run_children:
        blocks_allowed.append('repeat')
    blocks_allowed.append('move')
    blocks_allowed.append('turn_left')
    blocks_allowed.append('turn_right')
    blocks_allowed.append('pick_marker')
    blocks_allowed.append('put_marker')

    benchmark_task_json = {}
    # Task type is decided from the program_type
    benchmark_task_json['task_type'] = program_type
    code_karelgym_format = ast_to_json_karelgym_format(code_ast)
    task_grids_list = convert_task_grid_into_json(task_grids, program_type)

    benchmark_task_json['examples'] = task_grids_list
    benchmark_task_json['solution'] = {"program_type": program_type,
                                       "program_json": code_karelgym_format}
    benchmark_task_json['num_examples'] = len(task_grids)
    benchmark_task_json['num_blocks_allowed'] = num_blocks
    # Allow full store of blocks
    benchmark_task_json['type_blocks_allowed'] = STORE

    return benchmark_task_json, marker_activity_flag


def generate_taskcode(code_type: dict, num_blocks: int,
                      whilecounter_min=GLOBAL_WHILECOUNTER_MIN,
                      whilecounter_max=GLOBAL_WHILECOUNTER_MAX,
                      minrows=GLOBAL_MINROWS, maxrows=GLOBAL_MAXROWS,
                      mincols=GLOBAL_MINCOLS, maxcols=GLOBAL_MAXCOLS,
                      maxiters=MAXITERS_TASKGRID, debug_flag=False):

    # generate a random code, task pair
    program_type = nprand.choice(list(DIST_PROGRAM_TYPE.keys()), p=list(DIST_PROGRAM_TYPE.values()))
    code_json, ifelse_flag = generate_one_code(code_type, num_blocks, program_type)

    if code_json is None:
        if debug_flag:
            print("Unable to fulfill size constraints of the code!")
        return None

    # TODO code is already converted to karelgym here. But core code format is also needed for generating codetask pair.
    # Done here because redunduncy is checked with the karegym format
    code_ast = json_to_ast(code_json)
    code_karelgym_format = ast_to_json_karelgym_format(code_ast)
    # Added program type
    code_karelgym_format = {"program_type": program_type, "program_json": code_karelgym_format}

    # check the validity of the code
    flag_code_redundancy = check_code_redundancy(
        Code.parse_json(code_karelgym_format))  # returns True if code is valid
    if not flag_code_redundancy:
        if debug_flag:
            print(
                "Invalid code generated. Discarding due invalid action sequence in Code.")
        return None

    if debug_flag:
        print("Generated code:", code_json)
        print("IfElse flag:", ifelse_flag)

    # generate a random task
    if ifelse_flag:
        num_grids = 2  # Always fix the number of grids to 2 if the code has ifelse as a child of RUN
    else:
        num_grids = nprand.choice(list(DIST_NUMGRIDS.keys()),
                                  p=list(DIST_NUMGRIDS.values()))

    if debug_flag:
        print("Number of grids required:", num_grids)

    # specify pwall and pmarker and number of blocks
    pwall = nprand.choice([0, 1, random.uniform(0.0001, 1)],
                          p=list(DIST_PWALL.values()))
    pmarker = nprand.choice([0, random.uniform(0.0001, 1)],
                            p=list(DIST_PMARKER.values()))

    code_json_final, task_grids = generate_task_grids(code_json, num_grids,
                                                      whilecounter_min,
                                                      whilecounter_max,
                                                      minrows, maxrows, mincols,
                                                      maxcols, pwall, pmarker,
                                                      DIST_MARKERCOUNTS,
                                                      maxiters=maxiters,
                                                      debug_flag=False)

    if task_grids is not None:
        # generate the karelgym json
        karelgym_json, marker_activity = convert_to_karelgym_format_json(
            code_json_final, task_grids, program_type=program_type)

        # Added again program type
        karelgym_code = karelgym_json["solution"]

        # # Returns true for redundant code
        # flag_delta_debugging = check_codetask_redundancy_and_delta(code=Code.parse_json(karelgym_code),
        #                                                            task=Task.parse_json(karelgym_json),
        #                                                            unwrap=True,
        #                                                            delta_only=True)
        flag_delta_debugging = False

        if not marker_activity:  # no marker based actions in the code; we check for shortest-path
            flag_shortestpath = check_shortest_path(code=Code.parse_json(karelgym_code), task=Task.parse_json(karelgym_json))
        else:  # marker based actions in the code present; so we do not run the shortest path pipeline
            flag_shortestpath = True
        flag_solvability = check_solvability(code=Code.parse_json(karelgym_code), task=Task.parse_json(karelgym_json))
        flag_coverage = compute_coverage(code=Code.parse_json(karelgym_code), task=Task.parse_json(karelgym_json))
        if flag_shortestpath and flag_solvability and flag_coverage == 1 and not flag_delta_debugging:
            return karelgym_json
        else:
            if debug_flag:
                print(
                    "Invalid task generated. Discarding due to shortest path pruning or solvability failure.")
            return None
    else:  # taskgrid returned None
        if debug_flag:
            print(
                "Invalid task generated. Discarded due to CRASH during symbolic execution.")
        return None


if __name__ == "__main__":
    pass
