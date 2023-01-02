import json
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module

from src.karel_codetask_scoring.execquality import \
    count_codeworld_quality_from_emulator_actions
from src.karel_emulator.code import Code
from src.karel_emulator.fast_emulator import FastEmulator, EmuTick
from src.karel_emulator.world import World
from src.karel_symexecution.decision_makers import DecisionMaker
from src.karel_symexecution.symworld import SymWorld
from src.karel_symexecution.utils.enums import Direction
from src.karel_symexecution.utils.quadrants import get_quadrant_from_position
from src.neural_code2task_syn.vocabularies import location2idx, decision2idx, \
    idx2decision


# TODO: also try the pytorch differentiation method
def get_gradient(reward, probabilities):
    grad_value = reward / (1e-6 + probabilities.data)
    return -grad_value


def pre_process_input(world: SymWorld,
                      actions: List[str],
                      ticks: List[EmuTick],
                      current_location: str,
                      ast: Code):
    # The output contains a one-hot encoding of the current location,
    # the values [0,1] for the quality indicators, and the value [0, 1] for coverage
    tensor = np.zeros(len(location2idx) + 6 + 1)

    tensor[location2idx[current_location]] = 1

    no_moves, no_turns, no_segments, \
    no_long_segments, no_pick_markers, no_put_markers = \
        count_codeworld_quality_from_emulator_actions(actions)

    n = max(world.rows, world.cols)

    no_moves /= (2 * n)
    no_turns /= n
    no_segments /= (n / 2)
    no_long_segments /= (n / 3)
    no_pick_markers /= n
    no_put_markers /= n

    no_moves = min(no_moves, 1)
    no_turns = min(no_turns, 1)
    no_segments = min(no_segments, 1)
    no_long_segments = min(no_long_segments, 1)
    no_pick_markers = min(no_pick_markers, 1)
    no_put_markers = min(no_put_markers, 1)

    tensor[len(location2idx)] = no_moves
    tensor[len(location2idx) + 1] = no_turns
    tensor[len(location2idx) + 2] = no_segments
    tensor[len(location2idx) + 3] = no_long_segments
    tensor[len(location2idx) + 4] = no_pick_markers
    tensor[len(location2idx) + 5] = no_put_markers

    nb_nodes = ast.total_count
    # Find all unique locations visited from code
    combined_ticks = []
    combined_set = set()
    for tick_task in [ticks]:
        ticks_set = set()
        for tick in tick_task:
            ticks_set.add(str(tick.location))
            combined_set.add(str(tick.location))
        combined_ticks.append(ticks_set)

    coverage = len(combined_set) / nb_nodes

    tensor[len(location2idx) + 6] = coverage

    # TODO: padding?
    from training import MAX_WORLD_SIZE
    tensor_dict = {
        "features": torch.from_numpy(tensor).float(),
        "symworld": world.to_tensor(MAX_WORLD_SIZE[0])
    }

    return tensor_dict


def get_features_size():
    return len(location2idx) + 6 + 1


def get_output_size():
    return len(decision2idx)


class IntelligentDecisionMaker(DecisionMaker):
    def __init__(self,
                 network: Module,
                 emulator: Optional[FastEmulator] = None,
                 has_buffer: bool = False,
                 ):
        self._emulator = emulator

        self.buffer = []
        self.network = network

        self.has_buffer = has_buffer
        self.current_example = None
        self.current_example_index = -1

    def set_emulator(self, emulator):
        self._emulator = emulator

    def set_current_example(self, example):
        self.current_example = example
        self.current_example_index = -1

    def binary_decision(self):
        inp = pre_process_input(
            self._emulator.state.world,
            self._emulator.state.actions,
            self._emulator.state.ticks,
            self._emulator.current_location,
            self._emulator.ast,
        )
        logits = self.network(inp)
        logits[[value for key, value in decision2idx.items()
                if 'binary' not in key]] = float("-inf")
        probabilities = F.softmax(logits)
        if self.current_example:
            self.current_example_index += 1
            decision = self.current_example[self.current_example_index]["decision"]
        else:
            decision = torch.multinomial(probabilities, 1).item()
        # TODO: mask int decision?
        if self.has_buffer:
            to_mem = {
                "symworld": self._emulator.state.world.to_json(),
                "actions": self._emulator.state.actions,
                "ticks": self._emulator.state.ticks,
                "current": self._emulator.current_location,
                "code": self._emulator.ast,
                "decision": f'binary:{decision}',
                "probabilities": probabilities,
                "reward": None,
            }

            self.buffer.append(to_mem)

        return int(idx2decision[decision].split(":")[1])

    def pick_int(self, from_, to, for_):
        inp = pre_process_input(
            self._emulator.state.world,
            self._emulator.state.actions,
            self._emulator.state.ticks,
            self._emulator.current_location,
            self._emulator.ast,
        )
        logits = self.network(inp)
        logits[[value for key, value in decision2idx.items()
                if for_ not in key]] = float("-inf")
        probabilities = F.softmax(logits)
        if self.current_example:
            self.current_example_index += 1
            decision = self.current_example[self.current_example_index]["decision"]
        else:
            decision = torch.multinomial(probabilities, 1).item()
        # TODO: mask binary?
        if self.has_buffer:
            to_mem = {
                "symworld": self._emulator.state.world,
                "actions": self._emulator.state.actions,
                "ticks": self._emulator.state.ticks,
                "current": self._emulator.current_location,
                "decision": f'{for_}:{decision}',
                "probabilities": probabilities,
                "reward": None,
            }

            self.buffer.append(to_mem)

        return int(idx2decision[decision].split(":")[1])

    def reset_buffer(self):
        self.buffer = []

    def populate_rewards(self, reward):
        for idx, entry in enumerate(self.buffer):
            if entry["reward"] is None:
                if idx == len(self.buffer) - 1:
                    entry["reward"] = reward
                else:
                    entry["reward"] = 0

    def compute_gradients(self):
        batch_reward = np.mean([entry["reward"] for entry in self.buffer])

        variables = [x["probabilities"] for x in self.buffer]
        grads = [get_gradient(batch_reward, x["probabilities"]) for x in self.buffer]

        return batch_reward, variables, grads

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def get_parameters(self):
        return self.network.parameters()

    def __deepcopy__(self, memodict={}):
        copy_ = IntelligentDecisionMaker(self.network,
                                         self._emulator,
                                         self.has_buffer)
        copy_.buffer = self.buffer
        copy_.current_example = self.current_example
        copy_.current_example_index = self.current_example_index
        return copy_


def _get_position_of_interest(row, col, dir, query):
    if 'front' in query or 'Front' in query:
        if dir == Direction.north:
            return row + 1, col
        elif dir == Direction.south:
            return row - 1, col
        elif dir == Direction.east:
            return row, col + 1
        elif dir == Direction.west:
            return row, col - 1
    elif 'left' in query or 'Left' in query:
        if dir == Direction.north:
            return row, col - 1
        elif dir == Direction.south:
            return row, col + 1
        elif dir == Direction.east:
            return row + 1, col
        elif dir == Direction.west:
            return row - 1, col
    elif 'right' in query or 'Right' in query:
        if dir == Direction.north:
            return row, col + 1
        elif dir == Direction.south:
            return row, col - 1
        elif dir == Direction.east:
            return row - 1, col
        elif dir == Direction.west:
            return row + 1, col
    elif 'markers' or 'Markers' in query:
        return row, col


class ImitatingDecisionMaker(DecisionMaker):
    def __init__(self,
                 emulator: FastEmulator,
                 pregrid: World):
        self._emulator = emulator
        self.pregrid = pregrid

        self.buffer = []

    def set_emulator(self, emulator):
        self._emulator = emulator

    def binary_decision(self):
        current_loc = self._emulator.current_location
        hero_row = self._emulator.state.world.heroRow
        hero_col = self._emulator.state.world.heroCol
        hero_dir = self._emulator.state.world.heroDir
        row_of_int, col_of_int = _get_position_of_interest(hero_row, hero_col,
                                                           hero_dir, current_loc)
        if 'IsClear' in current_loc:
            decision = self.pregrid.blocked[row_of_int, col_of_int] == 0
        elif 'markers' in current_loc or 'Markers' in current_loc:
            current_orig_markers = self._emulator.state.world.original_markers[
                row_of_int, col_of_int]
            if current_orig_markers == 0.5:
                current_orig_markers = 1
            pregrid_markers = self.pregrid.markers[row_of_int, col_of_int]
            decision = pregrid_markers > int(current_orig_markers)
        else:
            print("ERROR: Unknown binary decision: {}".format(current_loc))
            raise NotImplementedError

        self.buffer.append(int(decision))

        return decision

    def pick_int(self, from_, to, for_):
        if 'dir' in for_:
            decision = self.pregrid.getHeroDirValue()
        elif 'pos' in for_:
            quad = get_quadrant_from_position(self.pregrid.heroRow,
                                              self.pregrid.heroCol,
                                              self.pregrid.rows,
                                              self.pregrid.cols)
            if quad == 'bottom_left':
                decision = 1
            elif quad == 'bottom_right':
                decision = 3
            elif quad == 'top_right':
                decision = 4
            elif quad == 'top_left':
                decision = 2
            elif quad == 'center':
                decision = 0
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.buffer.append(decision)

        return decision

    def save_buffer(self, path, code: Code):
        with open(path, 'a') as f:
            to_write = {
                "code": code.getJson(),
                "buffer": self.buffer,
                "rows": self.pregrid.rows,
                "cols": self.pregrid.cols,
            }
            f.write(json.dumps(to_write))

    def reset_buffer(self):
        self.buffer = []

    def __deepcopy__(self, memodict={}):
        copy_ = ImitatingDecisionMaker(self._emulator,
                                       self.pregrid)
        return copy_
