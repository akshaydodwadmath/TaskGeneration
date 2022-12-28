from src.karel_emulator.code import Code
from src.karel_emulator.executor import Executor
from src.karel_emulator.fast_emulator import FastEmulator, EmuResult
from src.karel_emulator.task import Task
from src.karel_emulator.world import World


def compute_coverage(code: Code, task: Task) -> float:
    """
    Given a single code-task pair returns coverage [0,1]
    """

    # Initialize executor
    executor = Executor()

    result = executor.execute(task, code)
    nb_nodes = code.total_count
    exec_info = [result['execution_info'] for result in result['emulator_result']]
    ticks_list = [execution_info["ticks"] for execution_info in exec_info]
    # Find all unique locations visited from code
    combined_ticks = []
    combined_set = set()
    for tick_task in ticks_list:
        ticks_set = set()
        for tick in tick_task:
            ticks_set.add(str(tick.location))
            combined_set.add(str(tick.location))
        combined_ticks.append(ticks_set)
    return len(combined_set) / nb_nodes


def compute_coverage_from_executor_result(result: dict, code: Code) -> float:
    nb_nodes = code.total_count
    exec_info = [result['execution_info'] for result in result['emulator_result']]
    ticks_list = [execution_info["ticks"] for execution_info in exec_info]
    # Find all unique locations visited from code
    combined_ticks = []
    combined_set = set()
    for tick_task in ticks_list:
        ticks_set = set()
        for tick in tick_task:
            ticks_set.add(str(tick.location))
            combined_set.add(str(tick.location))
        combined_ticks.append(ticks_set)
    return len(combined_set) / nb_nodes


def compute_coverage_from_emulator_result(result: EmuResult, code: Code) -> float:
    nb_nodes = code.total_count
    # Find all unique locations visited from code
    combined_ticks = []
    combined_set = set()
    for tick_task in [result.ticks]:
        ticks_set = set()
        for tick in tick_task:
            ticks_set.add(str(tick.location))
            combined_set.add(str(tick.location))
        combined_ticks.append(ticks_set)
    return len(combined_set) / nb_nodes
