from src.datagen.taskgen.get_code_rollout import generate_rollout
from src.datagen.taskgen.get_task_grid import get_one_task_grid


def generate_task_grids(code_json: dict, num_grids: int,
                        whilecounter_min: int, whilecounter_max: int,
                        minrows: int, maxrows: int, mincols: int, maxcols: int,
                        pwall: float, pmarker: float, markerdist: dict, maxiters,
                        debug_flag=False):
    task_grids = []
    for i in range(num_grids):
        # obtain the code rollout
        rollout = generate_rollout(code_json, whilecounter_min, whilecounter_max)
        if debug_flag:
            print("Rollout:", i)
            print(rollout)

        for j in range(maxiters):
            # generate a task grid
            grid = get_one_task_grid(rollout,
                                     minrows, maxrows, mincols, maxcols, pwall, pmarker,
                                     markerdist, debug_flag=debug_flag)
            if grid is not None:
                task_grids.append(grid)
                break

    if len(task_grids) == num_grids:
        return code_json, task_grids
    else:
        return code_json, None


if __name__ == "__main__":

    NUM_GRID = 2
    WHILECOUNTER_MIN = 2
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
                          {'type': 'while(bool_path_ahead)',
                           'children': [
                               {'type': 'move'},
                               {'type': 'put_marker'}
                           ]}

                      ]}

    # obtain the code rollout
    code = codedict_type2
    code_json, task_grid = generate_task_grids(code, NUM_GRID, WHILECOUNTER_MIN,
                                               WHILECOUNTER_MAX,
                                               MINROWS, MAXROWS, MINCOLS, MAXCOLS,
                                               PWALL, PMARKER, MARKERDIST)
    if task_grid is None:
        print("No valid task generated!")
    else:
        print("Code:", code_json)
        print("Generated the following task grid:", len(task_grid))
        print("-" * 40)
        for i, ele in enumerate(task_grid):
            print("Grid:", i)
            print(ele)
            print("-" * 20)
