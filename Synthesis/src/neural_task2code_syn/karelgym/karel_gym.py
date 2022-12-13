# Basic Karel Gym Wrapper on top of Karel State that can act on the current state using the Karel Compiler module. Holds
# the observation in the original AST representation(dictionary)

import json
import random

import numpy as np
from gym import Env, spaces
from tqdm import tqdm

from src.neural_task2code_syn.karelgym.karel_state import KarelState


class KarelGym(Env):
    """Karel OpenAI Gym Environment Wrapper. Basic Functionality of a Gym Environment is
        satisfied, i.e step, reset and render(print) functionality.
        The step function of the Env returns: obs, reward, Done, info.
        Observation is the original AST representation of the current src. This AST is maintained is JSON format.
        The reset function initializes the current task and chooses and new task for the collection of all the tasks.
        Returns:
        Environment Object that contains multiple Karel Tasks.
       """

    def __init__(self, grammar_use=True,
                 data_path="../../datasets/synthetic/iclr18_data_in_karelgym_format_10k/"
                           "train.json"):
        self.TASKS = []  # List of KarelState objects
        self.blocks_used = set()
        self.num_actions = 42  # See Action_Map in KarelCompiler/actions.py
        self.list_actions = np.arange(0, self.num_actions)
        self.H = 35  # Horizon
        self.action_space = spaces.Discrete(self.num_actions)
        self.max_pad = 18
        self.grammar_use = grammar_use
        self.KState_id = None
        # Load Karel States
        self._load_tasks(data_path)
        self.KState = self.sample_task()  # Initialize current task
        num_of_ios = len(self.KState.TRAIN_TASK.pregrids)
        self.observation_space = spaces.Dict(
            {"in_grids": spaces.Box(high=1, low=0,
                                    shape=[1, num_of_ios, 16, self.max_pad,
                                           self.max_pad],
                                    dtype=np.int32),
             "out_grids": spaces.Box(high=1, low=0,
                                     shape=[1, num_of_ios, 16, self.max_pad,
                                            self.max_pad],
                                     dtype=np.int32),
             "code": spaces.Dict()})
        self.steps_taken = 0
        self.done = False  # code done finishes the episode
        self.true_success = False  # True success is used for the reward design

    # Load tasks and target code and always heldout last task in pregrids and postgrids
    def _load_tasks(self, path):
        with open(path) as f:
            for example in tqdm(f):
                json_example = json.loads(example)
                self.TASKS.append(
                    (KarelState(json_example, enable_grammar=self.grammar_use)))

    def sample_task(self):
        self.KState_id = random.randint(0,
                                        len(self.TASKS) - 1)  # Comment randint returns [a,b]
        self.KState = self.TASKS[self.KState_id]
        return self.KState

    # Check successful execution in all grids and if size and type constraints are met
    def _get_reward(self):
        # True success comes from Karel state
        if self.true_success:
            return 1
        else:
            return 0

    # Step has to return obs, reward, done, info
    def step(self, action):
        self.steps_taken += 1
        self.blocks_used.add(action)
        cur_code, cur_world, cur_execution_info, body_left_empty, size_constraint_met, type_constraint_met, \
        self.done, self.true_success, semantic_success, solved = self.KState.step(
            ls_actions=[action])
        rew = self._get_reward()
        info = {"cur_code": cur_code, "cur_world": cur_world,
                "cur_exec_info": cur_execution_info,
                "body_left_empty": body_left_empty,
                "size_constraint": size_constraint_met,
                "type_constraint": type_constraint_met,
                "code_done": self.done,
                "true_success": self.true_success,
                "semantic_success": semantic_success,
                'solved': solved}
        if self.steps_taken == self.H:  # Horizon condition
            self.done = True
        return {"in_grids": self.KState.TRAIN_TASK.pregrids,
                "out_grids": self.KState.TRAIN_TASK.postgrids,
                "code": cur_code.getJson()}, rew, self.done, info

    def _pick_action(self):
        action_mask = self.KState.get_valid_actions()
        action = random.choices(population=self.list_actions, weights=action_mask)[0]
        return action

    def reset(self, *kwargs):
        # Reset current task to initial state
        self.steps_taken = 0
        self.blocks_used = set()
        self.done = False
        self.true_success = False
        # Choose a new task and update KState variable
        self.KState = self.sample_task()
        self.KState.reset()
        return {"in_grids": self.KState.TRAIN_TASK.pregrids,
                "out_grids": self.KState.TRAIN_TASK.postgrids,
                "code": self.KState.current_code.getJson()}

    def render(self, mode="human"):
        self.KState.TASK.pprint()
