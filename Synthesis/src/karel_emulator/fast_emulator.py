import copy

import numpy as np

from src.karel_emulator.tokens import actions, conditionals


class EmuLocationTuple(object):
    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __str__(self):
        return "{0}:{1}".format(self.name, self.index)


class EmuLocation(object):
    def __init__(self, tuples):
        self.tuples = tuples

    def add(self, name, index):
        tuples = copy.deepcopy(self.tuples)
        tuples.append(EmuLocationTuple(name, index))
        return EmuLocation(tuples)

    def __str__(self):
        return " ".join([str(x) for x in self.tuples])


class EmuTick(object):
    def __init__(self, location, type, value):
        self.location = location
        self.type = type
        self.value = value

    # def class_dict(self):
    #     return {"location":(self.location., "type":self.type}
    def __str__(self):
        return f"Location:{self.location}, type:{self.type}, value:{self.value}"

    def __repr__(self):
        return self.__str__()


class EmuResult(object):
    def __init__(self, status, inpgrid, outgrid, ticks, actions, crashed, timeout,
                 visited):
        self.status = status
        self.inpgrid = inpgrid
        self.outgrid = outgrid
        self.ticks = ticks
        self.actions = actions
        self.crashed = crashed
        self.timeout = timeout
        self.visited = visited


class FastEmuException(BaseException):
    def __init__(self, status):
        self.status = status


class EmuState(object):
    def __init__(self, world, max_ticks, max_actions):
        self.world = world
        self.max_ticks = max_ticks
        self.max_actions = max_actions
        self.crashed = False
        self.ticks = []
        self.actions = []

    def add_action(self, location, type):
        action_index = len(self.actions)
        self.__add_tick(EmuTick(location, 'action', action_index))
        self.actions.append(type)

    def add_condition_tick(self, location, result):
        self.__add_tick(EmuTick(location, "condition", result))

    def add_repeat_tick(self, location, index):
        self.__add_tick(EmuTick(location, 'repeat', index))

    def __add_tick(self, tick):
        if self.max_ticks is not None and \
                self.max_ticks != -1 and \
                len(self.ticks) >= self.max_ticks:
            raise FastEmuException('MAX_TICKS')
        self.ticks.append(tick)

    def __add_action(self, action):
        if self.max_actions is not None and \
                self.max_actions != -1 and \
                len(self.actions) >= self.max_actions:
            raise FastEmuException('MAX_ACTIONS')
        self.actions.append(action)


class FastEmulator(object):
    def __init__(self, max_ticks=None, max_actions=None):
        self.max_ticks = max_ticks
        self.max_actions = max_actions

        self.state = None
        self.current_location = None
        self.ast = None

        self.visited = None

        self.action_hash = {}
        for x in actions:
            self.action_hash[x] = 1

        self.conditional_hash = {}
        for x in conditionals:
            self.conditional_hash[x] = 1

    def emulate(self, ast, inpgrid):
        j_ast = ast.getJson()
        self.ast = ast
        original_inpgrid = copy.deepcopy(inpgrid)
        world = inpgrid
        self.state = EmuState(world, self.max_ticks, self.max_actions)
        location = EmuLocation([])

        self.visited = np.zeros((world.rows, world.cols), dtype=int)
        self.visited[world.heroRow, world.heroCol] = 1

        status = 'OK'
        try:
            assert (j_ast["type"] == "run")
            self.__emulate_block(j_ast, 'body', location, self.state)
        except FastEmuException as e:
            status = e.status

        crashed = status == 'CRASHED'
        timeout = status in ["MAX_TICKS", "MAX_ACTIONS"]
        result = EmuResult(status, original_inpgrid, self.state.world, self.state.ticks,
                           self.state.actions, crashed, timeout, self.visited)

        return result

    def __emulate_condition(self, condition, location, state):
        type = condition
        assert not isinstance(type, dict), \
            "Conditions are strings only, not dictionaries (i.e., no 'type': " \
            f"condition is needed). Problem condition: {condition}"
        if type not in self.conditional_hash:
            raise Exception("Type not supported: {0}".format(type))
        if type == 'noMarkersPresent':
            result = not state.world.markersPresent()
        elif type == "markersPresent" or type == 'boolGoal':
            result = state.world.markersPresent()
        elif type == "frontIsClear" or type == "rightIsClear" or type == "leftIsClear":
            conditional_func = getattr(state.world, type)
            result = conditional_func()
        elif type == "notFrontIsClear":
            conditional_func = getattr(state.world, "frontIsClear")
            result = not conditional_func()
        elif type == "notLeftIsClear":
            conditional_func = getattr(state.world, "leftIsClear")
            result = not conditional_func()
        elif type == "notRightIsClear":
            conditional_func = getattr(state.world, "rightIsClear")
            result = not conditional_func()
        else:
            raise Exception("Type not supported: {0}".format(type))
        state.add_condition_tick(location, result)
        return result

    def __emulate_block(self, parent, relationship, location, state):
        block = parent[relationship]
        for st_idx, node in enumerate(block):
            child_location = location.add(relationship, st_idx)
            type = node['type']
            if type in self.action_hash:
                self.current_location = type
                action_func = getattr(state.world, type)
                action_func()
                if type == 'move':
                    self.visited[self.state.world.heroRow][self.state.world.heroCol] \
                        += 1
                state.add_action(child_location, type)
                if state.world.isCrashed():
                    raise FastEmuException('CRASHED')
            elif type == 'cursor':
                pass
            elif type == 'repeat':
                times = node['times']
                for i in range(times):
                    self.current_location = f'repeat:{node["times"]}'
                    state.add_repeat_tick(child_location, i)
                    self.__emulate_block(node, 'body', child_location, state)

            elif type == 'while':
                while True:
                    self.current_location = f'while:{node["condition"]}'
                    res = self.__emulate_condition(node['condition'], child_location,
                                                   state)
                    if not res:
                        break
                    self.__emulate_block(node, 'body', child_location, state)

            elif type == 'repeatUntil':
                while True:
                    assert (node['condition'] == 'boolGoal')
                    self.current_location = f'repeatUntil:{node["condition"]}'
                    res = self.__emulate_condition(node['condition'], child_location,
                                                   state)
                    if res:  # This is the opposite of while
                        break
                    self.__emulate_block(node, 'body', child_location, state)

            elif type == 'if':
                self.current_location = f'if:{node["condition"]}'
                if self.__emulate_condition(node['condition'], child_location, state):
                    self.__emulate_block(node, 'body', child_location, state)
            elif type == 'ifElse':
                self.current_location = f'ifElse:{node["condition"]}'
                if self.__emulate_condition(node['condition'], child_location, state):
                    self.__emulate_block(node, 'ifBody', child_location, state)
                else:
                    self.__emulate_block(node, 'elseBody', child_location, state)
            else:
                raise Exception("Unknown type: {0}".format(type))
