import copy
from unittest.mock import MagicMock

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

from src.karel_emulator.code import Code
from src.karel_emulator.fast_emulator import FastEmulator
from src.karel_emulator.world import World
from src.karel_symexecution.decision_makers import RandomDecisionMaker
from src.karel_symexecution.symworld import SymWorld
from src.karel_symexecution.utils.enums import Direction


def unknown_only_closing(blocked: np.ndarray, unknowns: np.ndarray) -> np.ndarray:
    shape = np.ones((3, 3), dtype=int)
    result = binary_dilation(blocked, iterations=1,
                             border_value=0)
    result = binary_erosion(result, shape, iterations=1,
                            border_value=1)
    result = unknowns & result
    return result


class PostProcessor:

    @staticmethod
    def symworld_to_world(symworl):
        pass

    @staticmethod
    def handle_initial_markers(symworld):
        initial_markers = symworld.original_markers
        initial_markers = np.abs(initial_markers)
        initial_markers = np.where(initial_markers == 0.5, 1, initial_markers)
        initial_markers = initial_markers.astype(int)
        symworld.original_markers = initial_markers


class EmptySpacePostProcessor(PostProcessor):
    @staticmethod
    def symworld_to_world(symworld: SymWorld) -> (World, World):
        super(EmptySpacePostProcessor, EmptySpacePostProcessor). \
            handle_initial_markers(symworld)
        inp_world = World(symworld.rows, symworld.cols,
                          symworld.orig_hero_row, symworld.orig_hero_col,
                          symworld.orig_hero_dir,
                          symworld.blocked,
                          symworld.original_markers)
        out_world = World(symworld.rows, symworld.cols,
                          symworld.heroRow, symworld.heroCol,
                          symworld.heroDir,
                          symworld.blocked,
                          symworld.markers)
        return inp_world, out_world


class BlockedPostProcessor(PostProcessor):
    @staticmethod
    def symworld_to_world(symworld: SymWorld) -> (World, World):
        super(BlockedPostProcessor, BlockedPostProcessor). \
            handle_initial_markers(symworld)
        inp_world = World(symworld.rows, symworld.cols,
                          symworld.orig_hero_row, symworld.orig_hero_col,
                          symworld.orig_hero_dir,
                          symworld.blocked + symworld.unknown,
                          symworld.original_markers)
        out_world = World(symworld.rows, symworld.cols,
                          symworld.heroRow, symworld.heroCol,
                          symworld.heroDir,
                          symworld.blocked + symworld.unknown,
                          symworld.markers)
        return inp_world, out_world


class MorphologicalPostProcessor(PostProcessor):
    @staticmethod
    def symworld_to_world(symworld: SymWorld) -> (World, World):
        super(MorphologicalPostProcessor, MorphologicalPostProcessor). \
            handle_initial_markers(symworld)
        aux = copy.deepcopy(symworld.unknown)
        aux = np.where(aux == 1, np.random.randint(0, 2, (symworld.rows,
                                                          symworld.cols)), 0)
        aux = unknown_only_closing(aux + symworld.blocked, symworld.unknown)
        inp_world = World(symworld.rows, symworld.cols,
                          symworld.orig_hero_row, symworld.orig_hero_col,
                          symworld.orig_hero_dir,
                          aux + symworld.blocked,
                          symworld.original_markers)
        out_world = World(symworld.rows, symworld.cols,
                          symworld.heroRow, symworld.heroCol,
                          symworld.heroDir,
                          aux + symworld.blocked,
                          symworld.markers)
        return inp_world, out_world


class CopyPostProcessor(PostProcessor):
    @staticmethod
    def symworld_to_world(symworld: SymWorld,
                          pregrid_to_copy: World,
                          postgrid_to_copy: World) -> (World, World):
        super(CopyPostProcessor, CopyPostProcessor). \
            handle_initial_markers(symworld)
        aux_blocked = copy.deepcopy(pregrid_to_copy.blocked)
        mask = symworld.unknown == 0
        aux_blocked[mask] = 0

        aux_post_markers = copy.deepcopy(postgrid_to_copy.markers)
        mask = symworld.unknown == 0
        aux_post_markers[mask] = 0

        aux_pre_markers = copy.deepcopy(pregrid_to_copy.markers)
        mask = symworld.unknown == 0
        aux_pre_markers[mask] = 0

        inp_world = World(symworld.rows, symworld.cols,
                          symworld.orig_hero_row, symworld.orig_hero_col,
                          symworld.orig_hero_dir,
                          aux_blocked + symworld.blocked,
                          aux_pre_markers + symworld.original_markers)
        out_world = World(symworld.rows, symworld.cols,
                          symworld.heroRow, symworld.heroCol,
                          symworld.heroDir,
                          aux_blocked + symworld.blocked,
                          aux_post_markers + symworld.markers)
        return inp_world, out_world


if __name__ == '__main__':
    code = {
        "program_type": "karel",
        "program_json": {
            "type": "run",
            "body": [
                {
                    "type": "while",
                    "condition": "noMarkersPresent",
                    "body": [
                        {
                            "type": "move"
                        },
                        {
                            "type": "if",
                            "condition": "rightIsClear",
                            "body": [
                                {
                                    "type": "turnRight"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }

    # code = {"program_type": "karel",
    #         "program_json": {"type": "run",
    #                          "body":
    #                              [{"type": "while", "condition": "noMarkersPresent",
    #                                "body": [{"type": "putMarker"},
    #                                         {"type": "move"},
    #                                         {"type": "turnLeft"}]}
    #                               ]}
    #         }

    decision_maker = RandomDecisionMaker(None)
    marker_decisions = [False for _ in range(15)] + [True] + \
                       [False for _ in range(12)] + [True] + \
                       [False for _ in range(12)] + [True] + \
                       [False for _ in range(8)] + [True] + \
                       [False for _ in range(5)] + [True]
    decision_maker.binary_decision = MagicMock(side_effect=marker_decisions)

    decision_world = SymWorld.hero_position_only_init(16, 16,
                                                      12, 2, Direction.east)

    decision_world.set_decision_maker(decision_maker)

    emulator = FastEmulator(max_ticks=1000, max_actions=None)
    res = emulator.emulate(Code.parse_json(code), decision_world)

    print(res.outgrid.draw())
    print()

    in_world, out_world = MorphologicalPostProcessor.symworld_to_world(
        res.outgrid)

    print(out_world.draw())
