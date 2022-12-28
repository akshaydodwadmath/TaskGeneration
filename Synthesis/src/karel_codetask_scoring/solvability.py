from src.karel_emulator.code import Code
from src.karel_emulator.executor import Executor
from src.karel_emulator.fast_emulator import EmuResult
from src.karel_emulator.task import Task


def check_solvability(code: Code, task: Task) -> bool:
    """
    Given a single code-task pair returns solvability(i.e., grids are solved for the given code) True/False
    """

    executor = Executor()

    result = executor.execute(task, code)
    return result['task_success']


def check_solvability_from_executor_result(result: dict) -> bool:
    return result['task_success']


def check_solvability_from_emulator_result(result: EmuResult) -> bool:
    return not result.crashed and not result.timeout
