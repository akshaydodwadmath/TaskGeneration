import copy
import copy
import heapq
import logging
import time
from statistics import mean
from typing import Optional, Tuple, List, Generator

import numpy as np
import wandb
from decouple import config
from tqdm import tqdm

from src.karel_codetask_scoring.finalscore import compute_evaluation_score, \
    compute_synthesis_score_faster
from src.karel_data_converters.converter_format_iclr18_to_karelgym import \
    iclr18_codejson_to_karelgym_codejson
from src.karel_emulator.code import Code
from src.karel_emulator.fast_emulator import FastEmulator
from src.karel_emulator.task import Task
from src.karel_emulator.tokens import blocktypes
from src.karel_emulator.world import World
from src.karel_symexecution.decision_makers import DecisionMaker, \
    RandomDecisionMaker
from src.karel_symexecution.post_processor import PostProcessor, \
    EmptySpacePostProcessor
from src.karel_symexecution.symworld import SymWorld
from src.neural_code2task_syn.decision_makers import IntelligentDecisionMaker

DEBUG = False
TIME_DEBUG = False

MAX_GRID_SIZE = (10, 10)

#API_KEY = config("API_KEY")

POOL_SIZE = 50


# TODO: use try/except for the error when no new world can be found

def avg(lst: Generator) -> float:
    lst = list(lst)
    if len(lst) == 0:
        return 0.0
    return mean(lst)


class TaskSynthesizer:
    def __init__(self,
                 code: Code,
                 ref_task: Optional[Task],
                 decision_maker: DecisionMaker,
                 post_processor: PostProcessor,
                 max_grid_size: Optional[Tuple[int, int]] = None,
                 max_iterations: int = 1000000):
        self.code = code
        self.ref_task = ref_task
        self.max_iterations = max_iterations
        self.decision_maker = decision_maker
        self.post_processor = post_processor

        if max_grid_size is None:
            self.max_grid_size = MAX_GRID_SIZE

        self.init_task = Task([], [], code.type)

    def synthesize(self, num_tasks_to_generate,
                   num_blocks_allowed: int,
                   type_blocks_allowed: str,
                   type_: str,
                   log_freq: Optional[int] = 1,
                   init_symworlds: Optional[List[SymWorld]] = None,
                   ignore_diversity: bool = False,
                   ignore_dissimilarity: bool = False,
                   ) -> Tuple[Task, List[int]]:
        assert type_ in ["karel", "hoc"]

        scores = []
        if init_symworlds is None:
            init_symworlds = []

        for i in range(num_tasks_to_generate):
            if i < len(init_symworlds):
                inp_world, out_world, score = self.synthesize_world(
                    init_world=init_symworlds[i],
                    log_freq=log_freq,
                    log_prefix=f"world_{i}",
                    ignore_diversity=ignore_diversity,
                    ignore_dissimilarity=ignore_dissimilarity)
            else:
                inp_world, out_world, score = self.synthesize_world(
                    log_freq=log_freq,
                    log_prefix=f"world_{i}",
                    ignore_diversity=ignore_diversity,
                    ignore_dissimilarity=ignore_dissimilarity)

            self.init_task = Task(self.init_task.pregrids + [inp_world],
                                  self.init_task.postgrids + [out_world],
                                  type_=type_,
                                  num_blocks_allowed=num_blocks_allowed,
                                  type_blocks_allowed=type_blocks_allowed,
                                  num_examples=len(
                                      self.init_task.pregrids) + 1)
            scores.append(score)

        return self.init_task, scores

    def synthesize_world(self,
                         init_world: SymWorld = None,
                         log_freq: Optional[int] = 1,
                         ignore_dissimilarity: bool = False,
                         ignore_diversity: bool = False,
                         log_prefix: str = None) -> Tuple[World,
                                                          World,
                                                          float]:
        emulator = FastEmulator(max_ticks=1000, max_actions=1000)

        if isinstance(self.decision_maker, IntelligentDecisionMaker):
            self.decision_maker.set_emulator(emulator)

        max_score = -1.0
        best_inp_world = None
        best_out_world = None
        best_info = None

        heap = []

        if log_prefix:
            log_prefix = f"{log_prefix}/"
        else:
            log_prefix = ""

        if TIME_DEBUG:
            time_log = []
            time_from_worlds = []
            time_world_conv = []
            time_score = []
            time_emulation = []
            time_solv = []
            time_qual = []
            time_div = []
            time_diss = []
            time_cov = []
            time_score_exec = []

        time_lst = []
        for i in range(self.max_iterations):
            start_all = time.time()
            if init_world is None:
                current_init_world = \
                    SymWorld.empty_init(self.max_grid_size[0],
                                        self.max_grid_size[1],
                                        decision_maker=self.decision_maker)
            else:
                current_init_world = copy.deepcopy(init_world)
                current_init_world.set_decision_maker(self.decision_maker)

            if TIME_DEBUG:
                start = time.time()
            result = emulator.emulate(self.code, current_init_world)
            if TIME_DEBUG:
                end = time.time()
                time_emulation.append(end - start)

            # IMPORTANT: due to deep copies, the original decision maker is not
            # affected by the emulation process, so it will take the same decisions
            # next time. Thus, we pass the decision maker which was actually used
            # back to the self.decision_maker
            self.decision_maker = result.outgrid.decision_maker
            if isinstance(self.decision_maker, IntelligentDecisionMaker):
                print(len(self.decision_maker.buffer))
                self.decision_maker.reset_buffer()

            if TIME_DEBUG:
                start = time.time()
            inp_world, out_world = self.post_processor.symworld_to_world(result.outgrid)
            # TODO: I think this if should be here
            if self.code.type == "hoc":
                out_world.heroDir = "any"
            if TIME_DEBUG:
                end = time.time()
                time_world_conv.append(end - start)

            if TIME_DEBUG:
                start = time.time()
            task = Task(self.init_task.pregrids + [inp_world],
                        self.init_task.postgrids + [out_world],
                        type_=self.code.type)
            if TIME_DEBUG:
                end = time.time()
                time_from_worlds.append(end - start)

            if TIME_DEBUG:
                start = time.time()
                score, info, solv_time, qual_time, diss_time, div_time, cov_time, \
                exec_time = compute_synthesis_score_faster(result, self.code,
                                                           task, self.ref_task,
                                                           ignore_dissimilarity=ignore_dissimilarity,
                                                           ignore_diversity=ignore_diversity
                                                           )
                time_solv.append(solv_time)
                time_qual.append(qual_time)
                time_div.append(div_time)
                time_diss.append(diss_time)
                time_cov.append(cov_time)
                time_score_exec.append(exec_time)

            else:
                score, info = compute_synthesis_score_faster(result,
                                                             self.code,
                                                             task,
                                                             self.ref_task,
                                                             ignore_diversity=ignore_diversity,
                                                             ignore_dissimilarity=ignore_dissimilarity)
            if TIME_DEBUG:
                end = time.time()
                time_score.append(end - start)

            if DEBUG:
                print(f"Hero row: {inp_world.heroRow}, hero col: {inp_world.heroCol}, "
                      f"hero dir: {inp_world.heroDir}, score: {score}")
                # print(inp_world.toString())
                # print()
                # print(out_world.toString())

            if score > max_score:
                max_score = score
                best_inp_world = inp_world
                best_out_world = out_world
                best_info = info
                best_sym = result.outgrid

            if len(heap) < POOL_SIZE:
                heapq.heappush(heap, (score, i, inp_world,
                                      out_world, info))
            else:
                # Equivalent to a push, then a pop, but faster
                _ = heapq.heappushpop(heap, (score, i, inp_world,
                                             out_world, info))

            time_lst.append(time.time() - start_all)

            if TIME_DEBUG:
                start = time.time()
            if log_freq and i % log_freq == 0:
                logging.info(f"Current best score: {max_score}")
                logging.info(f"Current best info: {best_info}")

                if not best_info:
                    if log_freq:
                        wandb.log({f"{log_prefix}score": max_score})
                    continue

                # loc_diss = avg(x['loc_diss'] for x in best_info['dissimilarity'])
                # dir_diss = avg(x['dir_diss'] for x in best_info['dissimilarity'])
                # grid_diss = avg(x['grid_diss'] for x in best_info['dissimilarity'])

                if log_freq:
                    loc_diss = best_info['dissimilarity'][-1]['loc_diss']
                    dir_diss = best_info['dissimilarity'][-1]['dir_diss']
                    grid_diss = best_info['dissimilarity'][-1]['grid_diss']

                    loc_div = best_info['diversity'][-1]['loc_div']
                    dir_div = best_info['diversity'][-1]['dir_div']
                    grid_div = best_info['diversity'][-1]['grid_div']

                    wandb.log({f"{log_prefix}score": max_score,
                               f"{log_prefix}moves": best_info["quality"][-1]['moves'],
                               f"{log_prefix}turns": best_info["quality"][-1]['turns'],
                               f"{log_prefix}segments": best_info["quality"][-1][
                                   'segments'],
                               f"{log_prefix}long_segments": best_info["quality"][-1][
                                   'long_segments'],
                               f"{log_prefix}pick_markers": best_info["quality"][-1][
                                   'pick_markers'],
                               f"{log_prefix}put_markers": best_info["quality"][-1][
                                   'put_markers'],
                               f"{log_prefix}coverage": best_info["coverage"],

                               f"{log_prefix}loc_diss": loc_diss,
                               f"{log_prefix}dir_diss": dir_diss,
                               f"{log_prefix}grid_diss": grid_diss,
                               f"{log_prefix}quality": best_info["quality"][-1][
                                   'quality'],

                               f"{log_prefix}loc_div": loc_div,
                               f"{log_prefix}dir_div": dir_div,
                               f"{log_prefix}grid_div": grid_div,

                               # f"{log_prefix}pregrid":
                               #     wandb.Html(f"<pre><code>"
                               #                f"{best_inp_world.toString()}"
                               #                f"</pre></code>"),
                               # f"{log_prefix}postgrid":
                               #     wandb.Html(f"<pre><code>"
                               #                f"{best_out_world.toString()}"
                               #                f"</pre></code>")

                               })
            if TIME_DEBUG:
                end = time.time()
                time_log.append(end - start)

        if TIME_DEBUG:
            print(f"Time log: {avg(time_log)}")
            print(f"Time from worlds: {avg(time_from_worlds)}")
            print(f"Time world conv: {avg(time_world_conv)}")
            print(f"Time score: {avg(time_score)}")
            print(f"    Time solv: {avg(time_solv)}")
            print(f"    Time qual: {avg(time_qual)}")
            print(f"    Time div: {avg(time_div)}")
            print(f"    Time diss: {avg(time_diss)}")
            print(f"    Time cov: {avg(time_cov)}")
            print(f"    Time exec: {avg(time_score_exec)}")
            print(f"    Time total score: "
                  f"{avg(time_solv) + avg(time_qual) + avg(time_div) + avg(time_diss) + avg(time_cov) + avg(time_score_exec)}")
            print(f"Time emulation: {avg(time_emulation)}")
            print(f"Time suspects: ",
                  avg(time_score) + avg(time_emulation) + avg(time_world_conv))
            print(f"Time lst: {avg(time_lst)}")
            print(f"Total time: {sum(time_lst)}")

        if log_freq:
            wandb.log({f"{log_prefix}total_time": sum(time_lst),
                       f"{log_prefix}avg_time": avg(time_lst)})

        score, evaluation = \
            compute_evaluation_score(self.code,
                                     Task(
                                         self.init_task.pregrids + [best_inp_world],
                                         self.init_task.postgrids + [best_out_world],
                                         type_=self.code.type),
                                     self.ref_task,
                                     ignore_diversity=ignore_diversity,
                                     ignore_dissimilarity=ignore_dissimilarity)

        if not any([evaluation['redundancy'] == 'NOT RESPECTED',
                    evaluation['solvability'] == 'NOT RESPECTED',
                    evaluation['shortest_path'] == 'NOT RESPECTED',
                    evaluation['coverage'] != 1.0]):
            return best_inp_world, best_out_world, score

        # print(best_inp_world.draw())
        # print()
        # print(best_inp_world.draw())
        # print("No solution found yet")

        for _, _, inp_world, out_world, info in heapq.nlargest(POOL_SIZE, heap):
            score, evaluation = \
                compute_evaluation_score(self.code,
                                         Task(
                                             self.init_task.pregrids + [inp_world],
                                             self.init_task.postgrids + [out_world],
                                             type_=self.code.type),
                                         self.ref_task)
            # print(inp_world.draw())
            # print()
            # print(out_world.draw())

            if not any([evaluation['redundancy'] == 'NOT RESPECTED',
                        evaluation['solvability'] == 'NOT RESPECTED',
                        evaluation['shortest_path'] == 'NOT RESPECTED',
                        evaluation['coverage'] != 1.0]):
                # print("Found solution")
                # print(inp_world.draw())
                # print()
                # print(out_world.draw())
                return inp_world, out_world, score

        raise ValueError("No valid world found")


def obtain_karel_saturation_score_for_code(code, max_iterations=100_000):
    assert code.type == 'karel'

    ref_task = Task([], [], 'karel')
    post_processor = EmptySpacePostProcessor()
    decision_maker = RandomDecisionMaker.auto_init()

    # TODO: add the booleans for

    score = [0.0]
    try:
        task_synthesizer = TaskSynthesizer(code, ref_task,
                                           decision_maker, post_processor,
                                           max_iterations=max_iterations)
        task, score = task_synthesizer.synthesize(1, 50, ','.join(blocktypes),
                                                  log_freq=None,
                                                  type_='karel',
                                                  init_symworlds=None,
                                                  ignore_diversity=True,
                                                  ignore_dissimilarity=True)
    except Exception:
        pass

    return score


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Synthesize worlds with the help of '
    #                                              'SymWorld.')
    # parser.add_argument('--run', type=int)
    # parser.add_argument('--task', type=int)
    # parser.add_argument('--num_worlds', type=int, default=1)
    #
    # args = parser.parse_args()
    #
    # read_json = json.load(open(os.path.dirname(os.path.realpath(__file__)) +
    #                            "/synth_examples.json"))
    # init_tasks = json.load(open(os.path.dirname(os.path.realpath(__file__)) +
    #                             "/init_tasks.json"))
    # # result_dict = json.load(open(os.path.dirname(os.path.realpath(__file__)) +
    # #                              "/synth_results.json"))
    # result_dict = {}
    #
    # for entry in read_json:
    #     if entry["id"] != args.task:
    #         continue
    #
    #     code_json = entry["code"]
    #     task_json = entry["ref_task"]
    #
    #     init_symworlds = []
    #     if str(entry["id"]) in init_tasks:
    #         for init_world in init_tasks[str(entry["id"])]['examples']:
    #             init_symworlds.append(SymWorld.parse_json(init_world['symworld_json']))
    #
    #     max_iterations = 100000
    #
    #     post_processor = "empty_space" if code_json["program_type"] == "karel" else \
    #         "blocked"
    #
    #     config = {
    #         'max_iterations': max_iterations,
    #         'max_grid_size': MAX_GRID_SIZE,
    #         'code': code_json,
    #         'ref_task': task_json,
    #         'decision_maker': 'random',
    #         'post_processor': post_processor
    #     }
    #
    #     group = f"TaskSyn_{args.num_worlds}_worlds_{entry['common_name']}" if \
    #         args.num_worlds > 1 else f"TaskSyn_one_world_{entry['common_name']}"
    #
    #     wandb.login(key=API_KEY)
    #     wandb.init(project="tasksyn",
    #                entity="machine_teaching",
    #                config=config,
    #                group=group,
    #                mode="disabled",
    #                name=f"run_{args.run}",
    #                reinit=True)
    #
    #     code = Code.parse_json(code_json)
    #     ref_task = Task.parse_json(task_json)
    #
    #     decision_maker = RandomDecisionMaker.auto_init()  # IntelligentDecisionMaker()
    #     if code_json['program_type'] == 'karel':
    #         post_processor = EmptySpacePostProcessor()
    #     else:
    #         post_processor = BlockedPostProcessor()
    #
    #     task_synthesizer = TaskSynthesizer(code, ref_task,
    #                                        decision_maker, post_processor,
    #                                        max_iterations=max_iterations)
    #     task, score = task_synthesizer.synthesize(args.num_worlds, 50, ','.join(
    #         blocktypes), log_freq=1, type_=code_json['program_type'],
    #                                               init_symworlds=init_symworlds)
    #
    #     result_dict[entry["id"]] = {
    #         "id": entry["id"],
    #         "common_name": entry["common_name"],
    #         "task": task.to_json(),
    #         "score": score
    #     }
    #
    #     json.dump(result_dict, open(f"synth_tasks/synth_results_task"
    #                                 f"{entry['common_name']}_ru"
    #                                 f"n{args.run}_worlds{args.num_worlds}.json", "w+"))
    #     wandb.finish()

    program_json2 = {"run": [{"body": [{"body": [{"body": [{"type": "move"},
                                                           {"condition": {
                                                               "type": "rightIsClear"},
                                                               "elseBody": [
                                                                   {
                                                                       "type": "turnLeft"}],
                                                               "ifBody": [
                                                                   {
                                                                       "type": "putMarker"}],
                                                               "type": "ifElse"}],
                                                  "times": 3, "type": "repeat"}],
                                        "condition": {"type": "frontIsClear"},
                                        "type": "if"}], "times": 3, "type": "repeat"},
                             {"type":
                                  "putMarker"},
                             {"body": [{"type": "turnRight"}], "times": 2,
                              "type": "repeat"}, {"type": "move"}]}

    code_json = iclr18_codejson_to_karelgym_codejson(program_json2)

    print(code_json)

    code = Code('karel', code_json)
    scores = []
    start = time.time()
    for _ in range(20):
        score = obtain_karel_saturation_score_for_code(code, 200)
        scores.append(score)
    end = time.time()
    print(f"Time taken: {end - start}")

    print(scores)
    print(np.std(scores))
