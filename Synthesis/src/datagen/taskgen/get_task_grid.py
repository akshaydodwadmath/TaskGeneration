import copy

from src.datagen.taskgen.class_grid import Grid
from src.datagen.taskgen.get_code_rollout import generate_rollout


def get_one_task_grid(rollout: list,
                      minrows: int, maxrows: int, mincols: int, maxcols: int,
                      pwall: float, pmarker: float, markerdist: dict, debug_flag=False):
    # generate task grid for the rollout
    empty_grid = Grid(rollout, minrows, maxrows, mincols, maxcols, pwall, pmarker,
                      markerdist, debug_flag=debug_flag)
    # execute the code on the grid
    flag = empty_grid.generate_grid()

    if flag is None:
        if debug_flag:
            print("Crash Info:", empty_grid._crashinfo)
        return None
    else:
        return empty_grid


if __name__ == "__main__":

    WHILECOUNTER_MIN = 6
    WHILECOUNTER_MAX = 10
    MINROWS = 4
    MAXROWS = 10
    MINCOLS = 4
    MAXCOLS = 10
    PWALL = 0.5
    PMARKER = 0.5
    MARKERDIST = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1,
                  9: 0.1, 10: 0.1}

    codedict_type1 = {'type': 'run',
                      'children': [
                          {'type': 'move'},
                          {'type': 'move'},
                          {'type': 'turn_left'},
                          {'type': 'put_marker'},
                          {'type': 'pick_marker'},
                          {'type': 'pick_marker'},
                          {'type': 'move'},
                          {'type': 'move'}
                      ]}

    codedict_type2 = {'type': 'run',
                      'children': [
                          {'type': 'while(bool_no_marker_present)',
                           'children': [
                               {'type': 'put_marker'},
                               {'type': 'move'},
                           ]}

                      ]}

    codedict_type3a = {'type': 'run',
                       'children': [
                           {'type': 'while(bool_no_marker_present)',
                            'children': [
                                {'type': 'ifelse(bool_path_ahead)',
                                 'children': [
                                     {'type': 'do', 'children': [
                                         {'type': 'move'}
                                     ]},
                                     {'type': 'else', 'children': [
                                         {'type': 'turn_left'}
                                     ]}
                                 ]}
                            ]}

                       ]}

    codedict_type3b = {'type': 'run',
                       'children': [
                           {'type': 'repeat(8)',
                            'children': [
                                {'type': 'ifelse(bool_marker_present)',
                                 'children': [
                                     {'type': 'do', 'children': [
                                         {'type': 'pick_marker'}
                                     ]},
                                     {'type': 'else', 'children': [
                                         {'type': 'put_marker'}
                                     ]}
                                 ]},
                                {'type': 'move'}
                            ]}

                       ]}

    # obtain the code rollout
    code = copy.deepcopy(codedict_type3a)
    code_rollout = generate_rollout(code, WHILECOUNTER_MIN, WHILECOUNTER_MAX)
    print("Code rollout:", code_rollout)
    task_grid = get_one_task_grid(code_rollout,
                                  MINROWS, MAXROWS, MINCOLS, MAXCOLS, PWALL, PMARKER,
                                  MARKERDIST, debug_flag=False)
    if task_grid is None:
        print("No valid task generated!")
    else:
        print("Code:", code)
        print("Generated the following task grid:")
        print("-" * 40)
        print(task_grid)
