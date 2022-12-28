# TODO remove from this directory?

END_BODY = {
    "type": 'endBody'}  ## No such block. It is written this way to keep block["type"] reference consistent.

MOVE = {"type": "move"}
TURN_LEFT = {"type": "turnLeft"}
TURN_RIGHT = {"type": "turnRight"}
PICK_MARKER = {"type": "pickMarker"}
PUT_MARKER = {"type": "putMarker"}

COND_MARKERS_PRESENT = {"type": 'markersPresent'}
COND_NO_MARKERS_PRESENT = {"type": 'noMarkersPresent'}
COND_LEFT_IS_CLEAR = {"type": 'leftIsClear'}
COND_RIGHT_IS_CLEAR = {"type": 'rightIsClear'}
COND_FRONT_IS_CLEAR = {"type": 'frontIsClear'}
COND_NOT_LEFT_IS_CLEAR = {"condition": {"type": 'leftIsClear'}, "type": "not"}
COND_NOT_RIGHT_IS_CLEAR = {"condition": {"type": 'rightIsClear'}, "type": "not"}
COND_NOT_FRONT_IS_CLEAR = {"condition": {"type": 'frontIsClear'}, "type": "not"}

WHILE_MARKERS_PRESENT = {"condition": COND_MARKERS_PRESENT, "body": [], "type": "while"}
WHILE_NO_MARKERS_PRESENT = {"condition": COND_NO_MARKERS_PRESENT, "body": [],
                            "type": "while"}
WHILE_LEFT_IS_CLEAR = {"condition": COND_LEFT_IS_CLEAR, "body": [], "type": "while"}
WHILE_RIGHT_IS_CLEAR = {"condition": COND_RIGHT_IS_CLEAR, "body": [], "type": "while"}
WHILE_FRONT_IS_CLEAR = {"condition": COND_FRONT_IS_CLEAR, "body": [], "type": "while"}
WHILE_NOT_LEFT_IS_CLEAR = {"condition": COND_NOT_LEFT_IS_CLEAR, "body": [],
                           "type": "while"}
WHILE_NOT_RIGHT_IS_CLEAR = {"condition": COND_NOT_RIGHT_IS_CLEAR, "body": [],
                            "type": "while"}
WHILE_NOT_FRONT_IS_CLEAR = {"condition": COND_NOT_FRONT_IS_CLEAR, "body": [],
                            "type": "while"}

IFELSE_MARKERS_PRESENT = {"condition": COND_MARKERS_PRESENT, "elseBody": [],
                          "ifBody": [], "type": "ifElse"}
IFELSE_NO_MARKERS_PRESENT = {"condition": COND_NO_MARKERS_PRESENT, "elseBody": [],
                             "ifBody": [], "type": "ifElse"}
IFELSE_LEFT_IS_CLEAR = {"condition": COND_LEFT_IS_CLEAR, "elseBody": [], "ifBody": [],
                        "type": "ifElse"}
IFELSE_RIGHT_IS_CLEAR = {"condition": COND_RIGHT_IS_CLEAR, "elseBody": [], "ifBody": [],
                         "type": "ifElse"}
IFELSE_FRONT_IS_CLEAR = {"condition": COND_FRONT_IS_CLEAR, "elseBody": [], "ifBody": [],
                         "type": "ifElse"}
IFELSE_NOT_LEFT_IS_CLEAR = {"condition": COND_NOT_LEFT_IS_CLEAR, "elseBody": [],
                            "ifBody": [], "type": "ifElse"}
IFELSE_NOT_RIGHT_IS_CLEAR = {"condition": COND_NOT_RIGHT_IS_CLEAR, "elseBody": [],
                             "ifBody": [], "type": "ifElse"}
IFELSE_NOT_FRONT_IS_CLEAR = {"condition": COND_NOT_FRONT_IS_CLEAR, "elseBody": [],
                             "ifBody": [], "type": "ifElse"}

IF_MARKERS_PRESENT = {"condition": COND_MARKERS_PRESENT, "body": [], "type": "if"}
IF_NO_MARKERS_PRESENT = {"condition": COND_NO_MARKERS_PRESENT, "body": [], "type": "if"}
IF_LEFT_IS_CLEAR = {"condition": COND_LEFT_IS_CLEAR, "body": [], "type": "if"}
IF_RIGHT_IS_CLEAR = {"condition": COND_RIGHT_IS_CLEAR, "body": [], "type": "if"}
IF_FRONT_IS_CLEAR = {"condition": COND_FRONT_IS_CLEAR, "body": [], "type": "if"}
IF_NOT_LEFT_IS_CLEAR = {"condition": COND_NOT_LEFT_IS_CLEAR, "body": [], "type": "if"}
IF_NOT_RIGHT_IS_CLEAR = {"condition": COND_NOT_RIGHT_IS_CLEAR, "body": [], "type": "if"}
IF_NOT_FRONT_IS_CLEAR = {"condition": COND_NOT_FRONT_IS_CLEAR, "body": [], "type": "if"}

REPEAT_1 = {"body": [], "times": 1, "type": "repeat"}
REPEAT_2 = {"body": [], "times": 2, "type": "repeat"}
REPEAT_3 = {"body": [], "times": 3, "type": "repeat"}
REPEAT_4 = {"body": [], "times": 4, "type": "repeat"}
REPEAT_5 = {"body": [], "times": 5, "type": "repeat"}
REPEAT_6 = {"body": [], "times": 6, "type": "repeat"}
REPEAT_7 = {"body": [], "times": 7, "type": "repeat"}
REPEAT_8 = {"body": [], "times": 8, "type": "repeat"}
REPEAT_9 = {"body": [], "times": 9, "type": "repeat"}
REPEAT_10 = {"body": [], "times": 10, "type": "repeat"}
REPEAT_11 = {"body": [], "times": 11, "type": "repeat"}
REPEAT_12 = {"body": [], "times": 12, "type": "repeat"}

ACTION_MAP = {
    0: END_BODY,
    ## Basic Actions
    1: MOVE,
    2: TURN_LEFT,
    3: TURN_RIGHT,
    4: PICK_MARKER,
    5: PUT_MARKER,
    ## WHILEs
    6: WHILE_MARKERS_PRESENT,
    7: WHILE_NO_MARKERS_PRESENT,
    8: WHILE_LEFT_IS_CLEAR,
    9: WHILE_RIGHT_IS_CLEAR,
    10: WHILE_FRONT_IS_CLEAR,
    11: WHILE_NOT_LEFT_IS_CLEAR,
    12: WHILE_NOT_RIGHT_IS_CLEAR,
    13: WHILE_NOT_FRONT_IS_CLEAR,
    ## IFs
    14: IF_MARKERS_PRESENT,
    15: IF_NO_MARKERS_PRESENT,
    16: IF_LEFT_IS_CLEAR,
    17: IF_RIGHT_IS_CLEAR,
    18: IF_FRONT_IS_CLEAR,
    19: IF_NOT_LEFT_IS_CLEAR,
    20: IF_NOT_RIGHT_IS_CLEAR,
    21: IF_NOT_FRONT_IS_CLEAR,
    ## IFElSEs
    22: IFELSE_MARKERS_PRESENT,
    23: IFELSE_NO_MARKERS_PRESENT,
    24: IFELSE_LEFT_IS_CLEAR,
    25: IFELSE_RIGHT_IS_CLEAR,
    26: IFELSE_FRONT_IS_CLEAR,
    27: IFELSE_NOT_LEFT_IS_CLEAR,
    28: IFELSE_NOT_RIGHT_IS_CLEAR,
    29: IFELSE_NOT_FRONT_IS_CLEAR,
    ## REPEATs
    30: REPEAT_1,
    31: REPEAT_2,
    32: REPEAT_3,
    33: REPEAT_4,
    34: REPEAT_5,
    35: REPEAT_6,
    36: REPEAT_7,
    37: REPEAT_8,
    38: REPEAT_9,
    39: REPEAT_10,
    40: REPEAT_11,
    41: REPEAT_12,
}

BASIC_ACTIONS = [
    MOVE,
    TURN_LEFT,
    TURN_RIGHT,
    PICK_MARKER,
    PUT_MARKER
]

BASIC_CONDITIONS = [
    COND_MARKERS_PRESENT,
    COND_NO_MARKERS_PRESENT,
    COND_LEFT_IS_CLEAR,
    COND_RIGHT_IS_CLEAR,
    COND_FRONT_IS_CLEAR,
]

INTERNAL_ACTIONS = [0]  ## END_BODY doesn't count towards block limit


def get_allowed_actions(end_body_allowed=False,
                        if_allowed=False,
                        ifelse_allowed=False,
                        while_allowed=False,
                        repeat_allowed=False,
                        put_marker_allowed=False, pick_marker_allowed=False):
    bitmap = [0 for _ in ACTION_MAP.keys()]
    for i in range(1, 4):
        bitmap[i] = 1
    if end_body_allowed:
        bitmap[0] = 1
    if pick_marker_allowed:
        bitmap[4] = 1
    if put_marker_allowed:
        bitmap[5] = 1
    if while_allowed:
        for i in range(6, 14):
            bitmap[i] = 1
    if if_allowed:
        for i in range(14, 22):
            bitmap[i] = 1
    if ifelse_allowed:
        for i in range(22, 30):
            bitmap[i] = 1
    if repeat_allowed:
        for i in range(30, 42):
            bitmap[i] = 1

    return bitmap
