import itertools
from typing import Tuple

import numpy as np

from src.karel_emulator.task import Task
from src.karel_emulator.world import World


def compute_location_dissimilarity(world: World, reference_world: World) -> int:
    """
    Given a world and a reference world, returns the location dissimilarity
    (i.e., {0, 1}).
    """

    quad = find_quadrant(world.heroCol, world.heroRow, world.rows)
    ref_quad = find_quadrant(reference_world.heroCol, reference_world.heroRow,
                             reference_world.rows)

    if quad == ref_quad:
        return 0
    else:
        return 1


def compute_direction_dissimilarity(world: World, reference_world: World) -> int:
    """
    Given a world and a reference world, returns the direction dissimilarity
    (i.e., {0, 1}).
    """

    if world.heroDir != reference_world.heroDir:
        return 1
    return 0


def compute_grid_cell_dissimilarities(world: World, reference_world: World) -> float:
    """
    Given a world and a reference world, returns the normalized Hamming distance (
    i.e., [0, 1]) between the two worlds.
    """
    blocked_distance = np.count_nonzero(world.blocked != reference_world.blocked)

    world_marker_presence = np.clip(world.markers, 0, 1)
    reference_marker_presence = np.clip(reference_world.markers, 0, 1)
    marker_distance = np.count_nonzero(world_marker_presence !=
                                       reference_marker_presence)

    # Difference with baseline is due to missing wall surrounding the grid. Can be
    # 'fixed' by adding 2 to the number of rows and 2 to the number of cols below.

    return 2 * (blocked_distance + marker_distance) / (world.rows *
                                                       world.cols)


# TODO: should we use the symbolic-paths dissimilarity as well?
# def compute_world_dissimilarity(world: World, reference_world: World) -> Tuple[float,
#                                                                                dict]:
#     """
#     Given a world and a reference world, returns the world dissimilarity (i.e.,
#     [0, 1]).
#     """
#     loc_diss = compute_location_dissimilarity(world, reference_world)
#     dir_diss = compute_direction_dissimilarity(world, reference_world)
#     grid_diss = compute_grid_cell_dissimilarities(world, reference_world)
#
#     return (loc_diss + dir_diss + grid_diss) / 3, {
#         'loc_diss': loc_diss,
#         'dir_diss': dir_diss,
#         'grid_diss': grid_diss
#     }

def compute_world_dissimilarity(pregrid: World, postgrid: World,
                                reference_world: World) -> Tuple[float, dict]:
    """
    Given a world and a reference world, returns the world dissimilarity (i.e.,
    [0, 1]).
    """
    loc_diss = compute_location_dissimilarity(pregrid, reference_world)
    dir_diss = compute_direction_dissimilarity(pregrid, reference_world)

    pregrid_diss = compute_grid_cell_dissimilarities(pregrid, reference_world)
    postgrid_diss = compute_grid_cell_dissimilarities(postgrid, reference_world)
    grid_diss = max(pregrid_diss, postgrid_diss)
    grid_diss = min(grid_diss, 1)

    return (loc_diss + dir_diss + grid_diss) / 3, {
        'loc_diss': loc_diss,
        'dir_diss': dir_diss,
        'grid_diss': grid_diss
    }


# def compute_intra_task_dissimilarity(task: Task, full: bool = False,
#                                      method: str = 'min') -> Tuple[float,
#                                                                    dict]:
#     """
#     Given a task, returns the intra-task dissimilarity (i.e., [0, 1]).
#     """
#
#     pregrid_scores = []
#     pregrid_info = []
#     if full:
#         for world1, world2 in itertools.combinations(task.pregrids, 2):
#             score, info = compute_world_dissimilarity(world1, world2)
#             pregrid_scores.append(score)
#             pregrid_info.append(info)
#     else:
#         world1 = task.pregrids[-1]
#         for world2 in task.pregrids[:-1]:
#             score, info = compute_world_dissimilarity(world1, world2)
#             pregrid_scores.append(score)
#             pregrid_info.append(info)
#
#     postgrid_scores = []
#     postgrid_info = []
#     if full:
#         for world1, world2 in itertools.combinations(task.postgrids, 2):
#             score, info = compute_world_dissimilarity(world1, world2)
#             postgrid_scores.append(score)
#             postgrid_info.append(info)
#     else:
#         world1 = task.postgrids[-1]
#         for world2 in task.postgrids[:-1]:
#             score, info = compute_world_dissimilarity(world1, world2)
#             postgrid_scores.append(score)
#             postgrid_info.append(info)
#
#     if method == 'min':
#         return min(pregrid_scores + postgrid_scores), {
#             'pregrid_info': pregrid_info,
#             'postgrid_info': postgrid_info
#         }
#     elif method == 'mean':
#         return (sum(pregrid_scores) / len(pregrid_scores) +
#                 sum(postgrid_scores) / len(postgrid_scores)) / 2, {
#                    'pregrid_info': pregrid_info,
#                    'postgrid_info': postgrid_info
#                }
#     else:
#         raise ValueError(f'Unknown method {method}')
#
#
# def compute_inter_task_dissimilarity(task: Task, reference_task: Task,
#                                      full: bool = False) -> Tuple[float,
#                                                                   dict]:
#     """
#     Given a task and a reference task, returns the inter-task dissimilarity
#     (i.e., [0, 1]).
#     :param full: if True, returns the full inter-task dissimilarity,
#     otherwise returns only the dissimilarity between the last task and the reference
#     tasks.
#     """
#
#     pregrid_scores = []
#     pregrid_info = []
#     if full:
#         for world1, world2 in itertools.product(task.pregrids, reference_task.pregrids):
#             score, info = compute_world_dissimilarity(world1, world2)
#             pregrid_scores.append(score)
#             pregrid_info.append(info)
#     else:
#         world1 = task.pregrids[-1]
#         for world2 in reference_task.pregrids:
#             score, info = compute_world_dissimilarity(world1, world2)
#             pregrid_scores.append(score)
#             pregrid_info.append(info)
#
#     postgrid_scores = []
#     postgrid_info = []
#     if full:
#         for world1, world2 in itertools.product(task.postgrids,
#                                                 reference_task.postgrids):
#             score, info = compute_world_dissimilarity(world1, world2)
#             postgrid_scores.append(score)
#             postgrid_info.append(info)
#     else:
#         world1 = task.postgrids[-1]
#         for world2 in reference_task.postgrids:
#             score, info = compute_world_dissimilarity(world1, world2)
#             postgrid_scores.append(score)
#             postgrid_info.append(info)
#
#     return (sum(pregrid_scores) / len(pregrid_scores) +
#             sum(postgrid_scores) / len(postgrid_scores)) / 2, {
#                'pregrid_info': pregrid_info,
#                'postgrid_info': postgrid_info
#            }
#
#
def compute_task_dissimilarity(task: Task, reference_task: Task,
                               full_task_dissimilarity: bool = False,
                               for_entry: int = None) -> Tuple[float,
                                                               list]:
    """
    Given a task and a reference task, returns the task dissimilarity
    (i.e., [0, 1]).
    """
    scores = []
    info = []
    if full_task_dissimilarity:
        for pre, post, ref in itertools.product(task.pregrids, task.postgrids,
                                                reference_task.pregrids):
            score, info_ = compute_world_dissimilarity(pre, post, ref)
            scores.append(score)
            info.append(info_)
    elif for_entry is not None:
        assert for_entry < task.num_examples
        scores = []
        info = []
        pre, post = task.pregrids[for_entry], task.postgrids[for_entry]
        for ref in reference_task.pregrids:
            score, info_ = compute_world_dissimilarity(pre, post, ref)
            scores.append(score)
            info.append(info_)
    else:
        pre, post = task.pregrids[-1], task.postgrids[-1]
        for ref in reference_task.pregrids:
            score, info_ = compute_world_dissimilarity(pre, post, ref)
            scores.append(score)
            info.append(info_)

    if len(scores) == 0:
        return 0, info

    return sum(scores) / len(scores), info


def compute_world_diversity(pregrid: World, postgrid: World,
                            ref_pregrid: World, ref_postgrid: World) -> Tuple[float,
                                                                              dict]:
    """
    Given a world and a reference world, returns the world diversity
    (i.e., [0, 1]).
    """
    if pregrid == ref_pregrid or postgrid == ref_postgrid:
        return 0.0, {"loc_div": 0, 'dir_div': 0, 'grid_div': 0}
    else:
        preloc_diss = compute_location_dissimilarity(pregrid, ref_pregrid)
        postloc_diss = compute_location_dissimilarity(postgrid, ref_postgrid)
        loc_diss = min(preloc_diss, postloc_diss)

        predir_diss = compute_direction_dissimilarity(pregrid, ref_pregrid)
        postdir_diss = compute_direction_dissimilarity(postgrid, ref_postgrid)
        dir_diss = min(predir_diss, postdir_diss)

        pregrid_diss = compute_grid_cell_dissimilarities(pregrid, ref_pregrid)
        postgrid_diss = compute_grid_cell_dissimilarities(postgrid, ref_postgrid)
        grid_diss = max(pregrid_diss, postgrid_diss)
        grid_diss = min(grid_diss, 1)

        return (loc_diss + dir_diss + grid_diss) / 3, {
            'loc_div': loc_diss,
            'dir_div': dir_diss,
            'grid_div': grid_diss
        }


def compute_task_diversity(task: Task,
                           full_task_diversity: bool = False,
                           for_entry: int = None) -> Tuple[float, list]:
    """
    Given a task, returns the task diversity (i.e., [0, 1]).
    """
    if task.num_examples == 1:
        return 1, [{'loc_div': 1, 'dir_div': 1, 'grid_div': 1}]
    elif full_task_diversity:
        scores = []
        info = []
        for (pre1, post1), (pre2, post2) in itertools.combinations(zip(task.pregrids,
                                                                       task.postgrids),
                                                                   2):
            score, info_ = compute_world_diversity(pre1, post1, pre2, post2)
            scores.append(score)
            info.append(info_)
        return min(scores), info
    elif for_entry is not None:
        assert for_entry < task.num_examples
        scores = []
        info = []
        pre, post = task.pregrids[for_entry], task.postgrids[for_entry]
        for pre_ref, post_ref in zip(task.pregrids[:for_entry] +
                                     task.pregrids[for_entry + 1:],
                                     task.postgrids[:for_entry] +
                                     task.postgrids[for_entry + 1:]):
            score, info_ = compute_world_diversity(pre, post, pre_ref, post_ref)
            scores.append(score)
            info.append(info_)
        return min(scores), info
    else:
        scores = []
        info = []
        pre, post = task.pregrids[-1], task.postgrids[-1]
        for pre_ref, post_ref in zip(task.pregrids[:-1], task.postgrids[:-1]):
            score, info_ = compute_world_diversity(pre, post, pre_ref, post_ref)
            scores.append(score)
            info.append(info_)
        return min(scores), info


def find_quadrant(x, y, gridsz):
    if (x <= gridsz / 2) and (y <= gridsz / 2):
        quad = "bottom_left"
    elif (x > gridsz / 2) and (y <= gridsz / 2):
        quad = "bottom_right"
    elif (x <= gridsz / 2) and (y > gridsz / 2):
        quad = "top_left"
    else:
        quad = "top_right"
    # ordering to check quadrants is important,
    # first check four corners then check center
    if 8 <= gridsz <= 10:
        if (x >= 3) and (x <= 4) and (y >= 3) and (y <= 4):
            quad = "center"
    elif 11 <= gridsz <= 12:
        if (x >= 4) and (x <= 7) and (y >= 4) and (y <= 7):
            quad = "center"
    elif 13 <= gridsz <= 14:
        if (x >= 5) and (x <= 9) and (y >= 5) and (y <= 9):
            quad = "center"
    elif 15 <= gridsz <= 16:
        if (x >= 5) and (x <= 11) and (y >= 5) and (y <= 11):
            quad = "center"
    elif gridsz >= 17:
        if (x >= 6) and (x <= 12) and (y >= 6) and (y <= 12):
            quad = "center"

    return quad
