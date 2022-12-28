import collections
from src.datagen.wrappers.config_one_taskcode import BASIC_KAREL_ACTION_BLOCKS, BASIC_HOC_ACTION_BLOCKS
from src.karel_data_converters.converter_format_karelgym_to_readable import karelcode_json_to_ast


def code_to_codetype(code: dict):
    """
    Computes codetype from a given code
    :param: json code
    :return: codetype integer
    """

    if code["program_type"] == "hoc":
        BASIC_ACTION_BLOCKS = BASIC_HOC_ACTION_BLOCKS
    else:
        BASIC_ACTION_BLOCKS = BASIC_KAREL_ACTION_BLOCKS
    # Class ASTNode requires json string
    root = karelcode_json_to_ast(code["program_json"])
    if root._depth == 1:
        return 0
    elif root._depth == 4:
        return 7

    queue = collections.deque([root])
    D_count = 0
    D_depths = []
    # Traverse code to count number of constructs and nesting
    while len(queue) > 0:
        for i in range(len(queue)):
            node = queue.popleft()
            # convert action nodes to A
            if node._type in BASIC_ACTION_BLOCKS:
                node._type = "A"
            elif node._type == "run":
                node._type = "run"
            elif node._type == 'do' or node._type == 'else':
                    pass
            else:
                node._type = "D"
                D_count += 1
                D_depths.append(node._depth)
            for child in node._children:
                # Keep track only the ifelse branch with the larger depth
                if child._type == "do":
                    track_depth = child._depth
                if child._type == "else" and child._depth <= track_depth:
                    pass
                else:
                    queue.append(child)

    if root._depth == 2 and D_count == 1:
        return 1
    elif root._depth == 2 and D_count == 2:
        return 2
    elif root._depth == 3 and D_count == 2:
        return 3
    elif root._depth == 3 and D_count == 3 and D_depths == [2, 1, 1]:
        return 4
    elif root._depth == 3 and D_count == 3 and D_depths == [1, 2, 1]:
        return 5
    elif root._depth == 3 and D_count == 4 or D_count == 5 or D_count == 6:  # D can be in both if/else branches
        return 6
    else:
        return None

