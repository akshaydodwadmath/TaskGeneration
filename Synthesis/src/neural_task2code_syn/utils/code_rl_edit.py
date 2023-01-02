# TODO remove from this directory

import json
import src.neural_task2code_syn.utils.actions as actions


class CodeEdit:

    def __init__(self, astJson=None):
        self.unique_count = 0
        if astJson:  # Try to parse the given Json.
            try:
                cursor, open_bodies, success = self._parse_json(astJson)
                assert success
                self.astJson = astJson
                self.cursor = cursor
                self.open_bodies = open_bodies
            except Exception as e:
                print(e)
                print("This is not a valid ast JSON)")
                raise e
        else:  # Create an empty src
            self.astJson = {"run": [{"type": "cursor"}]}
            self.cursor = self.astJson["run"]
            self.open_bodies = []

    def is_done(self):
        return self.cursor is None

    def getJson(self):
        return self.astJson

    def _parse_json(self, astJson):
        '''
        Checks if astJson in the required format and extracts cursor and open bodies information.
        '''
        cursors_list = []  # Used as a static variable and populated by _parse_body with the path to cursor when cursor found

        def _parse_body(block_body, cursor_valid, cursor_path):
            '''
            A recursive helper function that parses the given body.
            block_body:list[dict] --> A block's body that potentially contains other blocks
            cursor_valid:bool --> True if block_body is on a valid path for sequential writing
            cursor_path:list[list] --> The path from root to this block
            '''
            if not cursor_path is None:
                cursor_path.append(block_body)
            for i in range(len(block_body)):
                child = block_body[i]
                cursor_valid_iter = cursor_valid and (i == len(
                    block_body) - 1)  # If this block is cursor invalid, all children will be too
                if child["type"] in ["while", "if"]:
                    assert len(child.keys()) == 3  # type, body, condition
                    assert _valid_condition(child["condition"])
                    assert _parse_body(child["body"], cursor_valid_iter,
                                       None if not cursor_valid_iter else cursor_path.copy())
                elif child["type"] == "repeat":
                    assert len(child.keys()) == 3  # type, body, times
                    assert isinstance(child["times"], int)
                    assert 0 < child["times"] < 13
                    assert _parse_body(child["body"], cursor_valid_iter,
                                       None if not cursor_valid_iter else cursor_path.copy())
                elif child["type"] == "ifElse":
                    assert len(child.keys()) == 4  # type, ifBody, elseBody, condition
                    assert _valid_condition(child["condition"])
                    if len(child["elseBody"]) == 0:
                        assert _parse_body(child["ifBody"], cursor_valid_iter,
                                           None if not cursor_valid_iter else cursor_path.copy() + [
                                               child["elseBody"]])
                    else:
                        assert _parse_body(child["ifBody"], False, None)
                        assert _parse_body(child["elseBody"], cursor_valid_iter,
                                           None if not cursor_valid_iter else cursor_path.copy())
                elif child in actions.BASIC_ACTIONS:
                    assert len(child.keys()) == 1
                else:
                    assert child[
                               "type"] == "cursor", f"Unknown block, found {child['type']}"
                    assert cursor_valid_iter
                    cursors_list.append(cursor_path)
            return True

        def _valid_condition(condition_block):
            '''
            Checks if the given block is a valid condition block
            '''
            if condition_block["type"] == "not":
                assert len(condition_block.keys()) == 2
                assert _valid_condition(condition_block["condition"])
            else:
                assert condition_block in actions.BASIC_CONDITIONS
            return True

        assert len(astJson.keys()) == 1  # Root should have only key 'run'
        assert _parse_body(astJson["run"], True, [])
        assert len(cursors_list) <= 1
        success = True
        cursor = cursors_list[0][-1] if len(cursors_list) == 1 else None
        open_bodies = cursors_list[0][:-1] if len(cursors_list) else []
        return cursor, open_bodies, success

    def toString(self):
        return json.dumps(self.astJson, sort_keys=True, indent=2)

    # def toPython(self, crashProtection = False):
    #     return PythonTranslator().toPython(self, crashProtection)

    def _copyConditionBlock(self, condition_block):
        inner_cond = condition_block.get("condition", None)
        if inner_cond:
            new_block = {"condition": {"type": inner_cond["type"]}, "type": "not"}
        else:
            new_block = {"type": condition_block["type"]}
        return new_block

    def take_action(self, actionid):
        '''
        actionid:Int --> An action id which would be mapped to a block using ACTION_MAP from actions.py

        Modifies the src according to given action.
        '''
        block = actions.ACTION_MAP[actionid]
        result_dict = {"code_done": False, "body_left_empty": False}
        last_block = self.cursor.pop()
        assert last_block["type"] == "cursor"

        if block["type"] == 'while':
            new_block = {"condition": self._copyConditionBlock(block["condition"]),
                         "body": [], "type": "while"}
            self.cursor.append(new_block)
            self.open_bodies.append(self.cursor)
            self.cursor = new_block["body"]

        elif block["type"] == 'repeat':
            new_block = {"body": [], "times": block["times"], "type": "repeat"}
            self.cursor.append(new_block)
            self.open_bodies.append(self.cursor)
            self.cursor = new_block["body"]

        elif block["type"] == 'ifElse':
            new_block = {"condition": self._copyConditionBlock(block["condition"]),
                         "elseBody": [], "ifBody": [], "type": "ifElse"}
            self.cursor.append(new_block)
            self.open_bodies.append(self.cursor)
            self.open_bodies.append(
                new_block["elseBody"])  # First thing to fill after ifBody is elseBody
            self.cursor = new_block["ifBody"]

        elif block["type"] == 'if':
            new_block = {"body": [],
                         "condition": self._copyConditionBlock(block["condition"]),
                         "type": "if"}
            self.cursor.append(new_block)
            self.open_bodies.append(self.cursor)
            self.cursor = new_block["body"]

        elif block["type"] == "endBody":
            if len(self.open_bodies) == 0:
                result_dict["code_done"] = True
                if len(self.cursor) == 0:
                    result_dict["body_left_empty"] = True
                result_dict["current_body_empty"] = False
                return result_dict
            elif len(self.cursor) == 0:
                result_dict["body_left_empty"] = True
                self.cursor = self.open_bodies.pop()
            else:
                self.cursor = self.open_bodies.pop()

        else:  # This should be the actionids 1-5 which are basic actions
            self.cursor.append({"type": block["type"]})

        self.cursor.append({"type": "cursor"})
        result_dict["current_body_empty"] = True if len(self.cursor) == 1 else False
        return result_dict

    def nb_nodes(self, blockJson):
        """
        Parses ast codes and returns number of unique nodes
        """
        if 'run' in blockJson:
            return self.nb_nodes(blockJson['run'])
        for block in blockJson:
            blockType = block['type']
            if blockType in ["move", "turnLeft", "turnRight", "pickMarker",
                             "putMarker"]:
                self.unique_count += 1

            # For loops
            elif blockType == 'repeat':
                self.unique_count += 1
                body = block['body']
                self.nb_nodes(body)

            # While loops
            elif blockType == 'while':
                self.unique_count += 1
                body = block['body']
                self.nb_nodes(body)

            # If statements
            elif blockType == 'if':
                self.unique_count += 1
                body = block['body']
                self.nb_nodes(body)

            # If/else statements
            elif blockType == 'ifElse':
                self.unique_count += 1
                ifBody = block['ifBody']
                elseBody = block['elseBody']
                self.nb_nodes(ifBody)
                self.nb_nodes(elseBody)

        return self.unique_count
