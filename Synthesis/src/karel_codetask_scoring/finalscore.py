import json
import time
from typing import Tuple

from src.karel_codetask_scoring.coverage import compute_coverage, \
    compute_coverage_from_executor_result, compute_coverage_from_emulator_result
from src.karel_codetask_scoring.deltadebugging import \
    check_codetask_redundancy_and_delta
from src.karel_codetask_scoring.execquality import compute_codetask_quality, \
    compute_codetask_quality_from_executor_result, \
    compute_codetask_quality_from_emulator_result, \
    compute_visitation_quality_from_executor_result
from src.karel_codetask_scoring.shortestpath import check_shortest_path
from src.karel_codetask_scoring.solvability import check_solvability, \
    check_solvability_from_executor_result, check_solvability_from_emulator_result
from src.karel_codetask_scoring.task_dissimilarity import compute_task_dissimilarity, \
    compute_task_diversity
from src.karel_emulator.code import Code
from src.karel_emulator.executor import Executor
from src.karel_emulator.fast_emulator import EmuResult
from src.karel_emulator.task import Task
from src.karel_emulator.tokens import actions

DELTA_QUALITY = 0.2

TIME_DEBUG = False


# TODO: maybe merge solvability, quality and coverage
def compute_final_score(code: Code, task: Task, reference_task: Task,
                        full_task_dissimilarity: bool = False) -> Tuple[float,
                                                                        dict]:
    """Compute the final score for a code snippet for a given task."""
    if not check_solvability(code, task):
        return 0.0, {}

    # TODO: this takes the most time
    # 0.086 seconds
    ########################################
    # if check_codetask_redundancy_and_delta(code, task,
    #                                        keep_empty_body=False,
    #                                        unwrap=True):
    #     return 0.0, {}
    ########################################

    if not check_shortest_path(code, task):
        return 0.0, {}

    score = 0.0
    norm = 0

    info = {}

    # 0.003 seconds
    ########################################
    aux_score, aux_info = compute_codetask_quality(code, task)
    score += aux_score
    norm += 1
    info['quality'] = aux_info

    if aux_score < DELTA_QUALITY:
        return 0.0, {}
    ########################################

    # 0.001 seconds
    ########################################
    if reference_task is not None and reference_task.num_examples > 0:
        aux_score, diss_info = compute_task_dissimilarity(task, reference_task,
                                                          full_task_dissimilarity)
        score += aux_score
        norm += 1
    else:
        diss_info = [{'loc_diss': 1,
                      'dir_diss': 1,
                      'grid_diss': 1}]
    info['dissimilarity'] = diss_info
    ########################################

    # 0.004 seconds
    ########################################
    aux_score = compute_coverage(code, task)
    score += aux_score
    norm += 1
    info['coverage'] = aux_score
    ########################################

    return score / norm, info


# TODO: two final score methods - one for training (coming up with worlds) and one
#  for testing (evaluating the quality of the worlds, which has delta debugging)
def compute_synthesis_score_fast(code: Code, task: Task, reference_task: Task,
                                 full_task_dissimilarity: bool = False):
    if TIME_DEBUG:
        solv_time = []
        qual_time = []
        cov_time = []
        diss_time = []
        div_time = []
        exec_time = []

    if TIME_DEBUG:
        start = time.time()
    executor = Executor()
    result = executor.execute(task, code)
    if TIME_DEBUG:
        exec_time.append(time.time() - start)

    if TIME_DEBUG:
        start = time.time()
    if not check_solvability_from_executor_result(result):
        if TIME_DEBUG:
            solv_time.append(time.time() - start)
            return 0.0, {}, solv_time[0], 0.0, 0.0, 0.0, 0.0, exec_time[0]

        return 0.0, {}
    if TIME_DEBUG:
        solv_time.append(time.time() - start)

    # if not check_shortest_path(code, task):
    #     return 0.0, {}

    score = 0.0
    norm = 0

    info = {}

    if TIME_DEBUG:
        start = time.time()
    quality_score, aux_info = \
        compute_codetask_quality_from_executor_result(result,
                                                      [max(world.rows, world.cols)
                                                       for world in task.pregrids],
                                                      task.type)
    if TIME_DEBUG:
        qual_time.append(time.time() - start)

    score += quality_score
    norm += 1
    info['quality'] = aux_info

    # if aux_score < DELTA_QUALITY:
    #     return 0.0, {}

    if reference_task is not None and reference_task.num_examples > 0:
        if TIME_DEBUG:
            start = time.time()
        dissimilarity_score, diss_info = \
            compute_task_dissimilarity(task,
                                       reference_task,
                                       full_task_dissimilarity)
        if TIME_DEBUG:
            diss_time.append(time.time() - start)
        score += dissimilarity_score
        norm += 1
    else:
        diss_info = [{'loc_diss': 1,
                      'dir_diss': 1,
                      'grid_diss': 1}]
    info['dissimilarity'] = diss_info

    if TIME_DEBUG:
        start = time.time()
    diversity_score, info_ = compute_task_diversity(task)
    if TIME_DEBUG:
        div_time.append(time.time() - start)
    score += diversity_score
    norm += 1
    info['diversity'] = info_

    if TIME_DEBUG:
        start = time.time()
    coverage_score = compute_coverage_from_executor_result(result, code)
    if TIME_DEBUG:
        cov_time.append(time.time() - start)
    score += coverage_score
    norm += 1
    info['coverage'] = coverage_score

    if TIME_DEBUG:
        return score / norm, info, solv_time[0], qual_time[0], diss_time[0], \
               div_time[0], \
               cov_time[0], exec_time[0]

    return score / norm, info


def compute_synthesis_score_faster(result: EmuResult, code: Code, task: Task,
                                   reference_task: Task,
                                   full_task_dissimilarity: bool = False,
                                   ignore_diversity: bool = False,
                                   ignore_dissimilarity: bool = False
                                   ):
    if TIME_DEBUG:
        solv_time = []
        qual_time = []
        cov_time = []
        diss_time = []
        div_time = []
        exec_time = []

    if TIME_DEBUG:
        start = time.time()
    # executor = Executor()
    # result = executor.execute(task, code)
    if TIME_DEBUG:
        exec_time.append(time.time() - start)

    if TIME_DEBUG:
        start = time.time()
    if not check_solvability_from_emulator_result(result):
        if TIME_DEBUG:
            solv_time.append(time.time() - start)
            return 0.0, {}, solv_time[0], 0.0, 0.0, 0.0, 0.0, exec_time[0]

        return 0.0, {}
    if TIME_DEBUG:
        solv_time.append(time.time() - start)

    # if not check_shortest_path(code, task):
    #     return 0.0, {}

    score = 0.0
    norm = 0

    info = {}

    if TIME_DEBUG:
        start = time.time()
    quality_score, aux_info = \
        compute_codetask_quality_from_emulator_result(result,
                                                      max(task.pregrids[-1].rows,
                                                          task.pregrids[-1].cols),
                                                      task.type)
    if TIME_DEBUG:
        qual_time.append(time.time() - start)

    score += quality_score
    norm += 1
    info['quality'] = aux_info

    # if aux_score < DELTA_QUALITY:
    #     return 0.0, {}

    if reference_task is not None and\
            not ignore_dissimilarity and \
            reference_task.num_examples > 0:
        if TIME_DEBUG:
            start = time.time()
        dissimilarity_score, diss_info = \
            compute_task_dissimilarity(task,
                                       reference_task,
                                       full_task_dissimilarity)
        if TIME_DEBUG:
            diss_time.append(time.time() - start)
        score += dissimilarity_score
        norm += 1
    else:
        diss_info = [{'loc_diss': -1,
                      'dir_diss': -1,
                      'grid_diss': -1}]
    info['dissimilarity'] = diss_info

    if not ignore_diversity:
        if TIME_DEBUG:
            start = time.time()
        diversity_score, info_ = compute_task_diversity(task)
        if TIME_DEBUG:
            div_time.append(time.time() - start)

        score += diversity_score
        norm += 1
        info['diversity'] = info_

    if TIME_DEBUG:
        start = time.time()
    coverage_score = compute_coverage_from_emulator_result(result, code)
    if TIME_DEBUG:
        cov_time.append(time.time() - start)
    score += coverage_score
    norm += 1
    info['coverage'] = coverage_score

    if not ignore_diversity and diversity_score == 0:
        score = 0

    if TIME_DEBUG:
        return score / norm, info, solv_time[0], qual_time[0], diss_time[0], \
               div_time[0], \
               cov_time[0], exec_time[0]

    return score / norm, info


def compute_evaluation_score(code: Code, task: Task, reference_task: Task,
                             full_task_dissimilarity: bool = False,
                             for_entry: int = None,
                             compute_shortest_path_quality: bool = False,
                             compute_visitation_quality: bool = False,
                             ignore_diversity: bool = False,
                             ignore_dissimilarity: bool = False) \
        -> Tuple[float, dict]:
    executor = Executor()
    result = executor.execute(task, code)

    info = {}

    if not check_solvability_from_executor_result(result):
        info['solvability'] = 'NOT RESPECTED'
    else:
        info['solvability'] = 'RESPECTED'

    # 0.086 seconds
    ########################################
    if check_codetask_redundancy_and_delta(code, task,
                                           keep_empty_body=False,
                                           unwrap=True):
        info['redundancy'] = 'NOT RESPECTED'
    else:
        info['redundancy'] = 'RESPECTED'
    ########################################

    simple_code = len(set([x for x in code.block_count
                           if code.block_count[x] != 0]) -
                      set(actions)) == 0

    score = 0.0
    norm = 0

    shortest_path, shortest_path_quality = check_shortest_path(code, task,
                                                               check_path_quality=True)
    if not shortest_path and not simple_code:
        info['shortest_path'] = 'NOT RESPECTED'
    else:
        info['shortest_path'] = 'RESPECTED'

    if compute_shortest_path_quality:
        score += sum(shortest_path_quality) / len(shortest_path_quality)
        norm += 1
        info['shortest_path_quality'] = shortest_path_quality

    if compute_visitation_quality:
        visitation_quality = compute_visitation_quality_from_executor_result(result)
        score += sum(visitation_quality) / len(visitation_quality)
        norm += 1
        info['visitation_quality'] = visitation_quality

    # 0.003 seconds
    ########################################
    aux_score, aux_info = \
        compute_codetask_quality_from_executor_result(result,
                                                      [max(world.rows, world.cols)
                                                       for world in task.pregrids],
                                                      task.type)
    score += aux_score
    norm += 1
    info['quality'] = aux_info

    if aux_score < DELTA_QUALITY:
        info['quality_delta'] = 'BELOW DELTA'
    else:
        info['quality_delta'] = 'ABOVE DELTA'
    ########################################

    # 0.001 seconds
    ########################################
    if not ignore_dissimilarity:
        if reference_task is not None and reference_task.num_examples > 0:
            aux_score, diss_info = compute_task_dissimilarity(task, reference_task,
                                                              full_task_dissimilarity,
                                                              for_entry)
            score += aux_score
            norm += 1
        else:
            diss_info = [{'loc_diss': 1,
                          'dir_diss': 1,
                          'grid_diss': 1}]
        info['dissimilarity'] = diss_info
    ########################################

    if not ignore_diversity:
        aux_score, info_ = compute_task_diversity(task,
                                                  full_task_dissimilarity,
                                                  for_entry)
        score += aux_score
        norm += 1
        info['diversity'] = info_

    # 0.004 seconds
    ########################################
    aux_score = compute_coverage_from_executor_result(result, code)
    score += aux_score
    norm += 1
    info['coverage'] = aux_score
    ########################################

    return score / norm, info


if __name__ == '__main__':
    json_results = json.load(open('../../synth_tasks'
                                  '/synth_results_taskH6_run50_worlds1.json',
                                  'r'))
    json_examples = json.load(open('../neural_code2task_syn/synth_examples.json', 'r'))

    for example in json_examples:
        code_json = example["code"]
        task_json = example["ref_task"]

        code = Code.parse_json(code_json)
        ref_task = Task.parse_json(task_json)

        if str(example["id"]) in json_results:
            print("Example {}".format(example["id"]))
            task_json = json_results[str(example["id"])]["task"]
            task = Task.parse_json(task_json)
            times = []
            # for i in range(10):
            #     start = time.time()
            #     score, info = compute_final_score_fast(code, task, ref_task)
            #     print(score)
            #     end = time.time()
            #     times.append(end - start)
            # print("Time: {}".format(np.mean(times)))

            for i, (pregrid, postgrid) in enumerate(zip(task.pregrids, task.postgrids)):
                # if i == 8:
                #     print("lol")
                score, info = compute_evaluation_score(code, task, ref_task,
                                                       for_entry=i,
                                                       compute_shortest_path_quality=
                                                       True,
                                                       compute_visitation_quality=True)
                print(info)
                print(i)
                print(score)
                print(pregrid.draw())
                print()
                print(postgrid.draw())
                print()
                print()
