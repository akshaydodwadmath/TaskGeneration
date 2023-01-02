import json

NUM_BLOCKS_ALLOWED = 1000
TYPE_BLOCKS_ALLOWED = "if,ifelse,while,repeat,putMarker,pickMarker"


def iclr18_codejson_to_karelgym_codejson(iclr18_code: json):
    ''' Converts a ICLR18 code JSON dictionary to a KarelGym code JSON.'''

    # dictionary to convert tokens
    bool_converter = {
        'markersPresent': 'markersPresent',
        'noMarkersPresent': 'noMarkersPresent',
        'frontIsClear': 'frontIsClear',
        'not_frontIsClear': 'notFrontIsClear',
        'leftIsClear': 'leftIsClear',
        'not_leftIsClear': 'notLeftIsClear',
        'rightIsClear': 'rightIsClear',
        'not_rightIsClear': 'notRightIsClear'
    }

    action_converter = {
        'move': 'move',
        'turnRight': 'turnRight',
        'turnLeft': 'turnLeft',
        'putMarker': 'putMarker',
        'pickMarker': 'pickMarker'
    }

    def get_children(json_node, if_type=False, else_type=False):
        if if_type:
            children = json_node.get('ifBody', [])
        elif else_type:
            children = json_node.get('elseBody', [])
        else:
            children = json_node.get('body', [])

        return children


    run_type = iclr18_code.get('run', None)
    if run_type is not None:
        node_type = "run"
        children = iclr18_code["run"]
    else:
        node_type = iclr18_code['type']
        children = get_children(iclr18_code)

    if node_type == "run":
        children_list = [iclr18_codejson_to_karelgym_codejson(c) for c in children]
        return {"type":'run', 'body':children_list}

    if node_type == 'ifElse':

        cond_json = iclr18_code["condition"]
        if cond_json['type'] == "not":
            cond_type = "not_" + cond_json['condition']['type']
        else:
            cond_type = cond_json['type']
            
        if cond_type == "not_noMarkersPresent":
            cond_type = "markersPresent"
        if cond_type == "not_markersPresent":
            cond_type = "noMarkersPresent"

        cond = bool_converter[cond_type]

        do_list = [iclr18_codejson_to_karelgym_codejson(child) for child in get_children(iclr18_code, if_type=True)]
        else_list = [iclr18_codejson_to_karelgym_codejson(child) for child in get_children(iclr18_code, else_type=True)]
        # node_type is 'maze_ifElse_isPathForward' or 'maze_ifElse_isPathLeft' or 'maze_ifElse_isPathRight'
        return { "type":"ifElse",
                 "condition": cond,
                "ifBody": do_list,
                "elseBody": else_list
        }

    elif node_type == 'if':

        cond_json = iclr18_code["condition"]
        if cond_json['type'] == "not":
            cond_type = "not_" + cond_json['condition']['type']
        else:
            cond_type = cond_json['type']
            
        if cond_type == "not_noMarkersPresent":
            cond_type = "markersPresent"
        if cond_type == "not_markersPresent":
            cond_type = "noMarkersPresent"

        cond = bool_converter[cond_type]

        do_list = [iclr18_codejson_to_karelgym_codejson(child) for child in get_children(iclr18_code)]

        return {
            "type":'if',
            "condition": cond,
            "body": do_list
        }



    elif node_type == 'while':

        while_list = [iclr18_codejson_to_karelgym_codejson(child) for child in children]
        cond_json = iclr18_code["condition"]
        if cond_json['type'] == "not":
            cond_type = "not_" + cond_json['condition']['type']
        else:
            cond_type = cond_json['type']
    
        if cond_type == "not_noMarkersPresent":
            cond_type = "markersPresent"
        if cond_type == "not_markersPresent":
            cond_type = "noMarkersPresent"

        cond = bool_converter[cond_type]
        return {
            "type": 'while',
            "condition": cond,
            "body": while_list
        }

    elif node_type == 'repeat':

        repeat_list = [iclr18_codejson_to_karelgym_codejson(child) for child in children]
        times = iclr18_code["times"]
        cond = times
        return {
           "type":'repeat',
           "times": cond,
           "body": repeat_list,
        }

    elif node_type == 'repeatUntil':

        repeat_until_goal_list = [iclr18_codejson_to_karelgym_codejson(child) for child in children]
        return {
            "type":'repeatUntil',
            "condition": 'boolGoal',
            "body": repeat_until_goal_list
        }

    elif node_type == 'move':
        return {"type":action_converter[node_type]}

    elif node_type == 'turnLeft':
        return {"type":action_converter[node_type]}

    elif node_type == 'turnRight':
        return {"type":action_converter[node_type]}

    elif node_type == 'pickMarker':
        return {"type":action_converter[node_type]}

    elif node_type == 'putMarker':
        return {"type":action_converter[node_type]}

    print('Unexpected node type, failing:', node_type)
    assert (False)



def convert_dataset(iclr_filepath, benchmark_filepath):
    """
    Converts dataset from original ICLR18 format to karelgym format
    """

    keys_to_add = ['inpgrid_json', 'outgrid_json']

    with open(iclr_filepath, 'r') as iclr_dataset:
        with open(benchmark_filepath, 'w') as benchmark_dataset:
            for line in iclr_dataset.readlines():
                sample = json.loads(line)
                true_sample = {
                    'examples': [{key: ex[key] for key in keys_to_add} for ex in
                                 sample['examples']],
                    'solution': {"program_type": "karel",
                                 "program_json": iclr18_codejson_to_karelgym_codejson(sample['program_json'])},
                    'num_examples': len(sample['examples']),
                    "num_blocks_allowed": NUM_BLOCKS_ALLOWED,
                    "type_blocks_allowed": TYPE_BLOCKS_ALLOWED}
                benchmark_dataset.write(json.dumps(true_sample))
                benchmark_dataset.write('\n')




# Example of dataset conversion from ICLR format to karelgym format
if __name__ == '__main__':
    # convert iclr18 program json into ASTNode
    program_json1 = {"run": [{"type": "pickMarker"}, {"type": "move"},
                            {"body": [{"body": [{"type": "putMarker"}], "times": 5, "type": "repeat"}],
                             "condition": {"type": "noMarkersPresent"}, "type": "while"}, {"type": "turnRight"}]}

    program_json2 = {"run": [{"body": [{"body": [{"body": [{"type": "move"},
                                                           {"condition": {"type": "rightIsClear"}, "elseBody": [{"type": "turnLeft"}], "ifBody": [{"type": "putMarker"}], "type": "ifElse"}], "times": 3, "type": "repeat"}], "condition": {"type": "frontIsClear"}, "type": "if"}], "times": 3, "type": "repeat"}, {"type":
 "putMarker"}, {"body": [{"type": "turnRight"}], "times": 2, "type": "repeat"}, {"type": "move"}]}


    print(iclr18_codejson_to_karelgym_codejson(program_json2))







    # convert_dataset("../../datasets/synthetic/iclr18_1m/test.json",
    #                 "../../datasets/synthetic/iclr18_data_in_karelgym_format_1m/test.json")

    # convert_dataset(
    #     "/AIML/misc/work/gtzannet/misc/GandRL_for_NPS/data/1m_6ex_karel/challenge/hoc.json",
    #     "/AIML/misc/work/gtzannet/karel-rl-benchmarks_code_adishs-github/datasets/realworld/"
    #     "iclr18_data_in_karelgym_format_hocmaze/test.json", data_type="realworld")
