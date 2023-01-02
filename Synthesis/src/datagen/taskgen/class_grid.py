import copy
import operator
import random

import numpy as np

from src.datagen.taskgen.get_code_rollout import RAst, unroll_ast, get_rollout_tokens
from src.datagen.wrappers.config_one_taskcode import GLOBAL_MAX_MARKERS_IN_CELL, \
    LOCATIONS_GENERATOR
from src.karel_symexecution.utils.enums import Quadrant
from src.karel_symexecution.utils.quadrants import get_position_from_quadrant

AVATAR_DIRECTIONS = ['east', 'north', 'west', 'south']

DIR_TO_COORD = {
    'east': (0, 1),
    'north': (-1, 0),
    'west': (0, -1),
    'south': (1, 0)
}

MAP_LEFT = {
    'east': 'north',
    'north': 'west',
    'west': 'south',
    'south': 'east'
}

MAP_RIGHT = {
    'east': 'south',
    'north': 'east',
    'west': 'north',
    'south': 'west'
}


def get_all_cells(rows, cols):
    return [(i, j) for i in range(rows) for j in range(cols)]


def get_all_quadrant_cells(rows, cols):
    if rows <= 2 and cols <= 2:
        return get_all_cells(rows, cols)
    else:
        return [get_position_from_quadrant(Quadrant.center, rows, cols),
                get_position_from_quadrant(Quadrant.top_left, rows, cols),
                get_position_from_quadrant(Quadrant.top_right, rows, cols),
                get_position_from_quadrant(Quadrant.bottom_left, rows, cols),
                get_position_from_quadrant(Quadrant.bottom_right, rows, cols)]


class Grid:
    def __init__(self, code_rollout: list, min_rows: int, max_rows: int, min_cols: int,
                 max_cols: int, pwalls: float, pmarkers: float,
                 markerdist: dict, max_markers=GLOBAL_MAX_MARKERS_IN_CELL,
                 debug_flag=False):
        self._code_rollout = code_rollout
        self._minrows = min_rows
        self._maxrows = max_rows
        self._mincols = min_cols
        self._maxcols = max_cols
        self._pwalls = pwalls
        self._pmarkers = pmarkers
        self._maxmarkers = max_markers
        self._markerdist = markerdist
        self._debug_flag = debug_flag
        self._lockedcells = {'pregrid': [], 'postgrid': []}
        self._markercells = {'pregrid': {},
                             'postgrid': {}}  # this attribute is a dictionary of dictionaries because we also need to save the number of markers in a grid-cell
        self._walls = {'pregrid': [], 'postgrid': []}
        self._emptycells = {'pregrid': [], 'postgrid': []}
        self._nrows, self._ncols, self._initloc_avatar, self._initdir_avatar = self.initialize_grid()
        self._currloc_avatar = self._initloc_avatar
        self._currdir_avatar = self._initdir_avatar
        self._crashinfo = {'flag': False}

    def initialize_grid(self):
        # decide the dimensions of the grid
        nrows = random.randint(self._minrows, self._maxrows)
        ncols = random.randint(self._mincols, self._maxcols)

        # # candidates for the initial location of the avatar
        # grid_midpoint = (int(nrows/2), int(ncols/2))
        # avatar_initlocs = [(0,0), (nrows-1, ncols-1), (0, ncols-1), (nrows-1, 0), grid_midpoint]
        # initloc = random.choice(avatar_initlocs)

        # randomly choose a starting location
        available_locations = eval(LOCATIONS_GENERATOR)(nrows, ncols)
        # initloc = (random.randint(0, nrows - 1), random.randint(0, ncols - 1))
        initloc = random.choice(available_locations)
        initdir = random.choice(AVATAR_DIRECTIONS)
        self._emptycells['pregrid'].append(initloc)
        self._emptycells['postgrid'].append(initloc)
        self._lockedcells['pregrid'].append(initloc)
        self._lockedcells['postgrid'].append(initloc)

        # for debugging
        if self._debug_flag:
            print("Dimensions of Grid:", nrows, ncols)
            print("INIT LOC/DIR of avatar:", initloc, initdir)

        return nrows, ncols, initloc, initdir

    # define the dynamics of the basic actions
    def move(self):
        next_loc = tuple(
            map(operator.add, self._currloc_avatar, DIR_TO_COORD[self._currdir_avatar]))
        flag_bounds, reason = self.check_bounds(next_loc)
        if not flag_bounds:
            self._crashinfo['coord'] = next_loc
            self._crashinfo['initloc_avatar'] = self._initloc_avatar
            self._crashinfo['initdir_avatar'] = self._initdir_avatar
            self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
            self._crashinfo['reason'] = reason
            self._crashinfo['flag'] = True
            return None

        if next_loc in self._walls['pregrid']:  # crash to WALL
            self._crashinfo['coord'] = next_loc
            self._crashinfo['initloc_avatar'] = self._initloc_avatar
            self._crashinfo['initdir_avatar'] = self._initdir_avatar
            self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
            self._crashinfo['reason'] = 'ERROR: CRASHED ON WALL'
            self._crashinfo['flag'] = True
            return None

        else:  # no crash
            self._currloc_avatar = next_loc
            if next_loc not in self._emptycells['pregrid']:
                self._emptycells['pregrid'].append(next_loc)
                self._lockedcells['pregrid'].append(next_loc)
            if next_loc not in self._emptycells['postgrid']:
                self._emptycells['postgrid'].append(next_loc)
                self._lockedcells['postgrid'].append(next_loc)

        if self._debug_flag:
            print("AFTER MOVE:", self)

        return 1

    def turn_left(self):
        self._currdir_avatar = MAP_LEFT[self._currdir_avatar]

        if self._debug_flag:
            print("AFTER TURN_LEFT:", self)
        return 1

    def turn_right(self):
        self._currdir_avatar = MAP_RIGHT[self._currdir_avatar]

        if self._debug_flag:
            print("AFTER TURN_RIGHT:", self)
        return 1

    def pick_marker(self):
        # add a marker to the pregrid
        loc_in_marker = self._markercells['pregrid'].get(self._currloc_avatar)
        if loc_in_marker is not None:
            self._markercells['pregrid'][
                self._currloc_avatar] += 1  # add a marker in the pregrid
        else:  # the loc does not have a marker yet
            if self._currloc_avatar in self._walls['pregrid']:
                self._crashinfo['coord'] = self._currloc_avatar
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: CRASHED ON WALL'
                self._crashinfo['flag'] = True
                return None

            self._markercells['pregrid'][
                self._currloc_avatar] = 1  # add the first marker in the pregrid location

        if self._debug_flag:
            print("AFTER PICK_MARKER:", self)
        return 1

    def put_marker(self):
        # add a marker to the postgrid
        loc_in_marker = self._markercells['postgrid'].get(self._currloc_avatar)
        if loc_in_marker is not None:
            self._markercells['postgrid'][
                self._currloc_avatar] += 1  # add the marker in the post-grid
        else:  # the loc does not have a marker yet
            if self._currloc_avatar in self._walls['postgrid']:
                self._crashinfo['coord'] = self._currloc_avatar
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: CRASHED ON WALL'
                self._crashinfo['flag'] = True
                return None

            self._markercells['postgrid'][
                self._currloc_avatar] = 1  # add the first marker in the postgrid location

        if self._debug_flag:
            print("AFTER PUT_MARKER:", self)
        return 1

    def check_bounds(self, loc):
        if loc[0] >= self._nrows:  # crash
            return False, 'ERROR: OUT OF BOUNDS'
        elif loc[1] >= self._ncols:  # crash
            return False, 'ERROR: OUT OF BOUNDS'
        elif loc[0] < 0:  # crash
            return False, 'ERROR: OUT OF BOUNDS'
        elif loc[1] < 0:  # crash
            return False, 'ERROR: OUT OF BOUNDS'
        else:
            return True, 'OKAY'

    def bool_path_ahead(self, val: str):
        ahead_loc = tuple(
            map(operator.add, self._currloc_avatar, DIR_TO_COORD[self._currdir_avatar]))
        flag_bounds, reason = self.check_bounds(ahead_loc)

        if val == 'true':  # path ahead is clear
            if not flag_bounds:
                self._crashinfo['coord'] = ahead_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = reason
                self._crashinfo['flag'] = True
                return None
            if ahead_loc in self._walls['pregrid']:
                self._crashinfo['coord'] = ahead_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                self._crashinfo['flag'] = True
                return None
            if ahead_loc in self._emptycells['pregrid']:
                pass
            elif ahead_loc in self._markercells['pregrid']:
                pass
            elif ahead_loc in self._markercells['postgrid']:
                pass
            else:  # the undecided cell has to be left empty
                self._emptycells['pregrid'].append(ahead_loc)
                if ahead_loc not in self._emptycells['postgrid']:
                    self._emptycells['postgrid'].append(ahead_loc)
        else:  # path ahead is NOT clear
            if not flag_bounds:  # path ahead is NOT clear because it is outside the grid
                pass
            else:
                if ahead_loc in self._emptycells['pregrid']:
                    self._crashinfo['coord'] = ahead_loc
                    self._crashinfo['initloc_avatar'] = self._initloc_avatar
                    self._crashinfo['initdir_avatar'] = self._initdir_avatar
                    self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                    self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                    self._crashinfo['flag'] = True
                    return None
                if ahead_loc in list(self._markercells['pregrid'].keys()):
                    self._crashinfo['coord'] = ahead_loc
                    self._crashinfo['initloc_avatar'] = self._initloc_avatar
                    self._crashinfo['initdir_avatar'] = self._initdir_avatar
                    self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                    self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                    self._crashinfo['flag'] = True
                    return None
                if ahead_loc not in self._walls['pregrid']:
                    self._walls['pregrid'].append(ahead_loc)
                if ahead_loc not in self._walls['postgrid']:
                    self._walls['postgrid'].append(ahead_loc)

        if self._debug_flag:
            print("AFTER BOOL_PATH_AHEAD:", self)
        return 1

    def bool_path_left(self, val: str):
        left_dir = MAP_LEFT[self._currdir_avatar]
        left_loc = tuple(
            map(operator.add, self._currloc_avatar, DIR_TO_COORD[left_dir]))
        flag_bounds, reason = self.check_bounds(left_loc)

        if val == 'true':  # path left is clear
            if not flag_bounds:
                self._crashinfo['coord'] = left_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = reason
                self._crashinfo['flag'] = True
                return None
            if left_loc in self._walls['pregrid']:
                self._crashinfo['coord'] = left_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                self._crashinfo['flag'] = True
                return None
            if left_loc in self._emptycells['pregrid']:
                pass
            elif left_loc in self._markercells['pregrid']:
                pass
            elif left_loc in self._markercells['postgrid']:
                pass
            else:  # the undecided cell has to be left empty
                self._emptycells['pregrid'].append(left_loc)
                if left_loc not in self._emptycells['postgrid']:
                    self._emptycells['postgrid'].append(left_loc)
        else:  # path left is NOT clear
            if not flag_bounds:
                pass
            else:
                if left_loc in self._emptycells['pregrid']:
                    self._crashinfo['coord'] = left_loc
                    self._crashinfo['initloc_avatar'] = self._initloc_avatar
                    self._crashinfo['initdir_avatar'] = self._initdir_avatar
                    self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                    self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                    self._crashinfo['flag'] = True
                    return None
                if left_loc in list(self._markercells['pregrid'].keys()):
                    self._crashinfo['coord'] = left_loc
                    self._crashinfo['initloc_avatar'] = self._initloc_avatar
                    self._crashinfo['initdir_avatar'] = self._initdir_avatar
                    self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                    self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                    self._crashinfo['flag'] = True
                    return None
                if left_loc not in self._walls['pregrid']:
                    self._walls['pregrid'].append(left_loc)
                if left_loc not in self._walls['postgrid']:
                    self._walls['postgrid'].append(left_loc)

        if self._debug_flag:
            print("AFTER BOOL_PATH_LEFT:", self)
        return 1

    def bool_path_right(self, val: str):
        right_dir = MAP_RIGHT[self._currdir_avatar]
        right_loc = tuple(
            map(operator.add, self._currloc_avatar, DIR_TO_COORD[right_dir]))
        flag_bounds, reason = self.check_bounds(right_loc)

        if val == 'true':  # path right is clear
            if not flag_bounds:
                self._crashinfo['coord'] = right_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = reason
                self._crashinfo['flag'] = True
                return None
            if right_loc in self._walls['pregrid']:
                self._crashinfo['coord'] = right_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                self._crashinfo['flag'] = True
                return None
            if right_loc in self._emptycells['pregrid']:
                pass
            elif right_loc in self._markercells['pregrid']:
                pass
            elif right_loc in self._markercells['postgrid']:
                pass
            else:  # the undecided cell has to be left empty
                self._emptycells['pregrid'].append(right_loc)
                if right_loc not in self._emptycells['postgrid']:
                    self._emptycells['postgrid'].append(right_loc)
        else:  # path right is NOT clear
            if not flag_bounds:
                pass
            else:
                if right_loc in self._emptycells['pregrid']:
                    self._crashinfo['coord'] = right_loc
                    self._crashinfo['initloc_avatar'] = self._initloc_avatar
                    self._crashinfo['initdir_avatar'] = self._initdir_avatar
                    self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                    self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                    self._crashinfo['flag'] = True
                    return None
                if right_loc in list(self._markercells['pregrid'].keys()):
                    self._crashinfo['coord'] = right_loc
                    self._crashinfo['initloc_avatar'] = self._initloc_avatar
                    self._crashinfo['initdir_avatar'] = self._initdir_avatar
                    self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                    self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                    self._crashinfo['flag'] = True
                    return None
                if right_loc not in self._walls['pregrid']:
                    self._walls['pregrid'].append(right_loc)
                if right_loc not in self._walls['postgrid']:
                    self._walls['postgrid'].append(right_loc)

        if self._debug_flag:
            print("AFTER BOOL_PATH_RIGHT:", self)
        return 1

    def bool_marker_present(self, val: str):
        curr_loc = self._currloc_avatar
        if val == 'true':  # marker present in the current loc
            if curr_loc in self._walls['pregrid']:
                self._crashinfo['coord'] = curr_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                self._crashinfo['flag'] = True
                return None
            markercount_pregrid = self._markercells['pregrid'].get(curr_loc)
            markercount_postgrid = self._markercells['postgrid'].get(curr_loc)
            if markercount_pregrid is None:  # add a marker
                self._markercells['pregrid'][curr_loc] = 1
            else:
                pass
            if markercount_postgrid is None:  # add a marker
                self._markercells['postgrid'][curr_loc] = 1
            else:
                pass

        else:  # no marker in current location
            markercount_pregrid = self._markercells['pregrid'].get(curr_loc)
            markercount_postgrid = self._markercells['postgrid'].get(curr_loc)
            if markercount_pregrid is not None:
                self._crashinfo['coord'] = curr_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                self._crashinfo['flag'] = True
                return None
            if markercount_postgrid is not None:
                self._crashinfo['coord'] = curr_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                self._crashinfo['flag'] = True
                return None

        if self._debug_flag:
            print("AFTER BOOL_MARKER_PRESENT:", self)
        return 1

    def bool_no_marker_present(self, val: str):
        curr_loc = self._currloc_avatar
        if val == 'false':  # marker present in the current loc
            if curr_loc in self._walls['pregrid']:
                self._crashinfo['coord'] = curr_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                self._crashinfo['flag'] = True
                return None
            markercount_pregrid = self._markercells['pregrid'].get(curr_loc)
            markercount_postgrid = self._markercells['postgrid'].get(curr_loc)
            if markercount_pregrid is None:  # add a marker
                self._markercells['pregrid'][curr_loc] = 1
            else:
                pass
            if markercount_postgrid is None:  # add a marker
                self._markercells['postgrid'][curr_loc] = 1
            else:
                pass

        else:  # no marker in current location
            markercount_pregrid = self._markercells['pregrid'].get(curr_loc)
            markercount_postgrid = self._markercells['postgrid'].get(curr_loc)
            if markercount_pregrid is not None:
                self._crashinfo['coord'] = curr_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                self._crashinfo['flag'] = True
                return None
            if markercount_postgrid is not None:
                self._crashinfo['coord'] = curr_loc
                self._crashinfo['initloc_avatar'] = self._initloc_avatar
                self._crashinfo['initdir_avatar'] = self._initdir_avatar
                self._crashinfo['grid_dims'] = (self._nrows, self._ncols)
                self._crashinfo['reason'] = 'ERROR: INCONSISTENT BOOL VAL'
                self._crashinfo['flag'] = True
                return None

        if self._debug_flag:
            print("AFTER BOOL_NO_MARKER_PRESENT:", self)
        return 1

    def postprocess_grid(self):
        all_coords = {
            'pregrid': [(i, j) for i in range(self._nrows) for j in range(self._ncols)]}
        all_coords['postgrid'] = copy.deepcopy(all_coords['pregrid'])

        emptycells_pregrid = self._emptycells['pregrid']
        emptycells_postgrid = self._emptycells['postgrid']
        markercells_pregrid = list(self._markercells['pregrid'].keys())
        markercells_postgrid = list(self._markercells['postgrid'].keys())
        walls_pregrid = self._walls['pregrid']
        walls_postgrid = self._walls['postgrid']
        locked_pregrid = markercells_pregrid + walls_pregrid + emptycells_pregrid
        locked_pregrid = list(set(locked_pregrid))
        locked_postgrid = emptycells_postgrid + walls_postgrid + markercells_postgrid
        locked_postgrid = list(set(locked_postgrid))

        all_coords['pregrid'] = list(set(all_coords['pregrid']) - set(locked_pregrid))
        all_coords['postgrid'] = list(
            set(all_coords['postgrid']) - set(locked_postgrid))

        # add the walls to the grid
        newwalls = [x for x in all_coords['pregrid'] if random.random() < self._pwalls]
        self._walls['pregrid'] += newwalls
        self._walls['pregrid'] = list(set(self._walls['pregrid']))
        self._walls['postgrid'] = copy.deepcopy(self._walls['pregrid'])
        all_coords['pregrid'] = list(set(all_coords['pregrid']) - set(newwalls))
        all_coords['postgrid'] = list(set(all_coords['postgrid']) - set(newwalls))

        # add the distractor markers to the grid
        newmarkers = [x for x in all_coords['pregrid'] if
                      random.random() < self._pmarkers]
        marker_elements = list(self._markerdist.keys())
        count_prob = list(self._markerdist.values())
        for ele in newmarkers:
            mcount = list(np.random.choice(marker_elements, 1, p=count_prob))[0]
            if self._markercells['pregrid'].get(ele) is not None:
                self._markercells['pregrid'][ele] += mcount
            else:
                self._markercells['pregrid'][ele] = mcount
            if self._markercells['postgrid'].get(ele) is not None:
                self._markercells['postgrid'][ele] += mcount
            else:
                self._markercells['postgrid'][ele] = mcount

        all_coords['pregrid'] = list(set(all_coords['pregrid']) - set(newmarkers))
        all_coords['postgrid'] = list(set(all_coords['postgrid']) - set(newmarkers))

        # add the remaining cells to empty
        self._emptycells['pregrid'] = self._emptycells['pregrid'] + all_coords[
            'pregrid']
        self._emptycells['pregrid'] = list(set(self._emptycells['pregrid']))

        self._emptycells['postgrid'] = self._emptycells['postgrid'] + all_coords[
            'postgrid']
        self._emptycells['postgrid'] = list(set(self._emptycells['postgrid']))

    def __repr__(self):

        pregrid = np.full([self._nrows, self._ncols], '?', dtype=str)
        postgrid = np.full([self._nrows, self._ncols], '?', dtype=str)

        # print("Walls:", self._walls)
        # print("Empty cells:", self._emptycells)
        # print("Markers:", self._markercells)

        for ele in self._walls['pregrid']:
            pregrid[ele[0], ele[1]] = '#'
            postgrid[ele[0], ele[1]] = '#'
        for ele in self._emptycells['pregrid']:
            pregrid[ele[0], ele[1]] = '.'
        for ele in self._emptycells['postgrid']:
            postgrid[ele[0], ele[1]] = '.'
        for key, val in self._markercells['pregrid'].items():
            pregrid[key[0], key[1]] = str(val)
        for key, val in self._markercells['postgrid'].items():
            postgrid[key[0], key[1]] = str(val)

        markerinavatarloc_conversion = {'>': 'E', '^': 'A', '<': 'Z',
                                        'v': 'V'}  # symbols to indicate that the avatar location also has markers

        # add the avatar in the pregrid
        if self._initdir_avatar == "east":
            pregrid[self._initloc_avatar[0], self._initloc_avatar[1]] = '>'
        elif self._initdir_avatar == "north":
            pregrid[self._initloc_avatar[0], self._initloc_avatar[1]] = '^'
        elif self._initdir_avatar == "west":
            pregrid[self._initloc_avatar[0], self._initloc_avatar[1]] = '<'
        else:  # south
            pregrid[self._initloc_avatar[0], self._initloc_avatar[1]] = 'v'
        if self._initloc_avatar in list(self._markercells['pregrid'].keys()):
            marker_count = self._markercells['pregrid'][self._initloc_avatar]
            pregrid[self._initloc_avatar[0], self._initloc_avatar[1]] = \
                markerinavatarloc_conversion[
                    pregrid[self._initloc_avatar[0], self._initloc_avatar[1]]]

        # add the avatar in the postgrid
        if self._currdir_avatar == "east":
            postgrid[self._currloc_avatar[0], self._currloc_avatar[1]] = '>'
        elif self._currdir_avatar == "north":
            postgrid[self._currloc_avatar[0], self._currloc_avatar[1]] = '^'
        elif self._currdir_avatar == "west":
            postgrid[self._currloc_avatar[0], self._currloc_avatar[1]] = '<'
        else:  # south
            postgrid[self._currloc_avatar[0], self._currloc_avatar[1]] = 'v'

        if self._currloc_avatar in list(self._markercells['postgrid'].keys()):
            marker_count = self._markercells['postgrid'][self._currloc_avatar]
            postgrid[self._currloc_avatar[0], self._currloc_avatar[1]] = \
                markerinavatarloc_conversion[
                    postgrid[self._currloc_avatar[0], self._currloc_avatar[1]]]

        return "Pregrid:\n" + np.array2string(
            pregrid) + '\nPostgrid:\n' + np.array2string(postgrid)

    # routine to execute the roll-out on the task-grid
    def execute_rollout(self):
        ret = None
        for ele in self._code_rollout:
            if ':' in ele:  # boolean val
                bool_var = ele.split(':')[0]
                bool_val = ele.split(':')[1]
                if bool_var == 'bool_path_ahead':
                    ret = self.bool_path_ahead(bool_val)
                elif bool_var == 'bool_path_left':
                    ret = self.bool_path_left(bool_val)
                elif bool_var == 'bool_path_right':
                    ret = self.bool_path_right(bool_val)
                elif bool_var == 'bool_marker_present':
                    ret = self.bool_marker_present(bool_val)
                elif bool_var == 'bool_no_marker_present':
                    ret = self.bool_no_marker_present(bool_val)
                else:
                    if self._debug_flag:
                        print("Unknown boolean token encountered!", bool_var)
                    ret = None
            else:
                if ele == 'run':
                    ret = 1
                elif ele == 'move':
                    ret = self.move()
                elif ele == 'turn_left':
                    ret = self.turn_left()
                elif ele == 'turn_right':
                    ret = self.turn_right()
                elif ele == 'pick_marker':
                    ret = self.pick_marker()
                elif ele == 'put_marker':
                    ret = self.put_marker()
                else:
                    if self._debug_flag:
                        print("Unknown token encountered!", ele)
                    ret = None

            if ret is None:
                return None

        return 1

    def generate_grid(self):
        ret = self.execute_rollout()
        if ret is None:
            # print("GENERATE_GRID STATUS: No valid task generated!")
            return None
        else:
            # post-process the grid
            self.postprocess_grid()
            # check if any of the cells have more markers than allowed
            pregrid_markers = list(self._markercells['pregrid'].values())
            postgrid_markers = list(self._markercells['postgrid'].values())
            if any(m > self._maxmarkers for m in pregrid_markers):
                return None
            elif any(m > self._maxmarkers for m in postgrid_markers):
                return None
            else:
                return 1


if __name__ == "__main__":

    example_depth1 = RAst('run', 'RUN', children=[

        RAst('move', '1'),
        RAst('turn_left', '2'),
        RAst('move', '3'),
        RAst('move', '4'),
        RAst('turn_right', '5')

    ])

    # obtain the code roll-out
    unrolled_ast = unroll_ast(example_depth1, while_counter_min=2, while_counter_max=5)
    print("After unrolling:", unrolled_ast)
    code_tokens = get_rollout_tokens(unrolled_ast)
    print("Code tokens:", code_tokens)

    # generate the task grid
    marker_dist = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1,
                   9: 0.1, 10: 0.1}
    empty_grid = Grid(code_tokens, 4, 6, 4, 6, 0.5, 0.5, marker_dist)
    # execute the code on the grid
    flag = empty_grid.generate_grid()
    if flag is None:
        print("No valid task grid generated!")
        print(empty_grid._crashinfo)
    else:
        print("Valid task generated!")
        print(empty_grid)
