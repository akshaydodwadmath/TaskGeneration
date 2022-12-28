import src.neural_task2code_syn.utils.actions as actions
from src.karel_emulator.code import Code
from src.karel_emulator.fast_emulator import FastEmulator
from src.karel_emulator.task import Task
from src.karel_emulator.world import World


class KarelState:

    def __init__(self, examples_json, enable_grammar, max_ticks=1000, max_actions=None):

        num_examples = int(examples_json['num_examples'])
        num_blocks_allowed = int(examples_json['num_blocks_allowed'])
        type_blocks_allowed_str = examples_json['type_blocks_allowed']
        examples = examples_json['examples']

        # Create the Task with all IO pairs
        pregrids = [World.parseJson(example['inpgrid_json']) for example in examples]
        postgrids = [World.parseJson(example['outgrid_json']) for example in examples]

        # Create Task with one held out IO pair. This Task is used for training.
        if len(pregrids) > 2 or len(
                postgrids) > 2:  # For Tasks with only 1 I/O pair there is no heldout
            pregrids_training = pregrids[:-1]
            postgrids_training = postgrids[:-1]
        else:
            pregrids_training = pregrids
            postgrids_training = postgrids

        # Get the allowed blocks as a list of integers(keys of Action Map)
        type_blocks_allowed_str_list = type_blocks_allowed_str.split(",")

        self.enable_grammar = enable_grammar
        if enable_grammar:
            self.type_blocks_allowed_empty_body = \
                actions.get_allowed_actions(end_body_allowed=False,
                                            if_allowed="if" in type_blocks_allowed_str_list,
                                            ifelse_allowed="ifelse" in type_blocks_allowed_str_list,
                                            while_allowed="while" in type_blocks_allowed_str_list,
                                            repeat_allowed="repeat" in type_blocks_allowed_str_list,
                                            pick_marker_allowed="pickMarker" in type_blocks_allowed_str_list,
                                            put_marker_allowed="putMarker" in type_blocks_allowed_str_list)

        type_blocks_allowed = \
            actions.get_allowed_actions(end_body_allowed=True,
                                        if_allowed="if" in type_blocks_allowed_str_list,
                                        ifelse_allowed="ifelse" in type_blocks_allowed_str_list,
                                        while_allowed="while" in type_blocks_allowed_str_list,
                                        repeat_allowed="repeat" in type_blocks_allowed_str_list,
                                        pick_marker_allowed="pickMarker" in type_blocks_allowed_str_list,
                                        put_marker_allowed="putMarker" in type_blocks_allowed_str_list)

        info = {"num_blocks_allowed": num_blocks_allowed,
                "type_blocks_allowed": type_blocks_allowed}
        self.TASK = Task(pregrids, postgrids, info)  # Inference Task
        self.TRAIN_TASK = Task(pregrids_training, postgrids_training,
                               info)  # Training Task with heldout pair
        self.current_code = Code()
        self.TARGET_CODE = examples_json['program_json']
        self.NUM_BLOCKS_ALLOWED = num_blocks_allowed
        self.TYPE_BLOCKS_ALLOWED = type_blocks_allowed
        self.CUR_BLOCKS_ALLOWED = None
        self.EMULATOR = FastEmulator(max_ticks=max_ticks, max_actions=max_actions)
        self.current_grid = self.TASK.pregrids
        self.current_execution_info = [{} for i in range(len(self.current_grid))]
        self.num_used_blocks = 0
        self.type_blocks_allowed_respected = []
        self.current_body_empty = False
        self.done = False

    def grammar_check(self):
        if self.enable_grammar and self.current_body_empty:
            self.CUR_BLOCKS_ALLOWED = self.type_blocks_allowed_empty_body
        else:
            self.CUR_BLOCKS_ALLOWED = self.TYPE_BLOCKS_ALLOWED

    # Step function receives a list of actions as input
    def step(self, ls_actions):

        result_dict = {}

        for actionid in ls_actions:

            # Break if ast code is done
            if self.done:
                break

            result_dict = self.current_code.take_action(
                actionid)  # Result dict is not resetted

            if actionid not in actions.INTERNAL_ACTIONS:
                self.num_used_blocks += 1

            self.grammar_check()
            if self.CUR_BLOCKS_ALLOWED[actionid] == 1:
                self.type_blocks_allowed_respected.append(1)
            else:
                self.type_blocks_allowed_respected.append(0)

            self.done = result_dict['code_done']

        result_dict_list = self.EMULATOR.execute(self.current_code, self.TASK)
        self.current_grid = [result['grid'] for result in result_dict_list]
        self.current_execution_info = [result['execution_info'] for result in
                                       result_dict_list]
        self.current_body_empty = result_dict['current_body_empty']

        solved = [execution_info['success'] for execution_info in
                  self.current_execution_info]

        true_success = all(solved) \
                       and not result_dict['body_left_empty'] \
                       and self.done \
                       and self.num_used_blocks <= self.NUM_BLOCKS_ALLOWED \
                       and all(self.type_blocks_allowed_respected) == 1

        solved = sum(solved) / len(solved)

        # For semantic success we evaluate the code on training IO pairs.
        semantic_dict_list = self.EMULATOR.execute(self.current_code, self.TRAIN_TASK)
        semantic_execution_info = [result['execution_info'] for result in
                                   semantic_dict_list]
        semantic_success = all(
            [execution_info['success'] for execution_info in semantic_execution_info]) \
                           and not result_dict['body_left_empty'] \
                           and self.done \
                           and self.num_used_blocks <= self.NUM_BLOCKS_ALLOWED \
                           and all(self.type_blocks_allowed_respected) == 1

        return self.current_code, self.current_grid, self.current_execution_info, \
               result_dict[
                   'body_left_empty'], self.num_used_blocks <= self.NUM_BLOCKS_ALLOWED, \
               self.type_blocks_allowed_respected, self.done, true_success, \
               semantic_success, solved

    # Reset Function
    def reset(self):
        self.current_code = Code()
        self.current_grid = self.TASK.pregrids
        self.current_execution_info = [{} for i in range(len(self.current_grid))]
        self.num_used_blocks = 0
        self.type_blocks_allowed_respected = []
        self.done = False

    # Returns binary mask of valid actions
    def get_valid_actions(self):
        if self.done:
            return []
        else:
            if self.enable_grammar and self.current_body_empty:
                return self.type_blocks_allowed_empty_body
            else:
                return self.TYPE_BLOCKS_ALLOWED

    def pprint(self, print_task=False):
        if print_task:
            self.TASK.pprint()
        print(self.current_code.toString())
        for grid in self.current_grid:
            print(grid.toString())


if __name__ == '__main__':
    # Example on loading a Karelstate object.
    # You can load any custom example json with the specified format.
    import json

    # Provide your json example path
    json_path = "../../datasets/realworld/data_format_language_train.json"
    with open(json_path) as f:
        # KarelState Object
        karelstate = KarelState(examples_json=json.load(f), enable_grammar=True)

    # See karelstate's target code
    print("Target code:", karelstate.TARGET_CODE)
    # See karelstate's task object (contains all pregrids/postgrids)
    print("Karel state's task object:", karelstate.TASK)  # Inference tasks
    # See Task object's pregrids or postgrids as world objects
    print("Karel state's task pregrids", karelstate.TASK.pregrids)
    print("Karel state's task train tasks",
          karelstate.TRAIN_TASK)  # Training tasks(last grid is heldout)

    # Take step or reset karelstate is also possible. Takes as input a list of action tokens
    step_output = karelstate.step(ls_actions=[4, 6, 7, 34])
    karelstate.reset()
    karelstate.pprint(print_task=True)
