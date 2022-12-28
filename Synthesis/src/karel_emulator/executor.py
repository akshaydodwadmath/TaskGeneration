import copy

from src.karel_emulator.code import Code
from src.karel_emulator.fast_emulator import FastEmulator
from src.karel_emulator.task import Task


class Executor:
    def __init__(self, max_ticks=1000, max_actions=None):
        self.EMULATOR = FastEmulator(max_ticks=max_ticks, max_actions=max_actions)

    def execute(self, T, C):

        result_list = []
        solved = []
        blocks_used = []
        for pregrid, postgrid in zip(T.pregrids, T.postgrids):
            result = self.EMULATOR.emulate(C, copy.deepcopy(pregrid))

            execution_info = {'status': result.status,
                              'success': result.outgrid == postgrid,
                              'ticks': result.ticks,
                              'actions': result.actions,
                              'crashed': result.crashed,
                              'timeout': result.timeout,
                              'visited': result.visited}

            result_dict = {'execution_info': execution_info, 'grid': result.outgrid}

            result_list.append(result_dict)
            solved.append(execution_info["success"] and not execution_info["timeout"])

        num_blocks_satisfied = C.total_count <= T.num_blocks_allowed  # Check if the max number of blocks is satisfied

        # Check if allowed blocks are used
        for key in C.block_count:
            if C.block_count[key] > 0:
                blocks_used.append(key)
        type_blocks_satified = blocks_used in T.type_blocks_allowed

        res = {"task_success": all(solved),
               "num_blocks_satisfied": num_blocks_satisfied,
               "type_blocks_satisfied": type_blocks_satified, "open_body": C.open_body,
               "emulator_result": result_list}

        return res


if __name__ == '__main__':
    import json

    with open("../../tests/karel_emulator/test_data/taskany.json", "r") as f:
        taskcode = json.load(f)

    with open("../../tests/karel_emulator/test_data/code_repeatuntil.json", "r") as c:
        code = json.load(c)

    t = Task.parse_json(taskcode)
    c = Code.parse_json(code)
    print(c)
    exec = Executor()
    result = exec.execute(t, c)
    print(result)
