nb_helper_tkns = 5

idx2tok = {
    0: "END_BODY",
    # Basic Actions
    1: "MOVE",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
    4: "PICK_MARKER",
    5: "PUT_MARKER",
    # WHILEs
    6: "WHILE_MARKERS_PRESENT",
    7: "WHILE_NO_MARKERS_PRESENT",
    8: "WHILE_LEFT_IS_CLEAR",
    9: "WHILE_RIGHT_IS_CLEAR",
    10: "WHILE_FRONT_IS_CLEAR",
    11: "WHILE_NOT_LEFT_IS_CLEAR",
    12: "WHILE_NOT_RIGHT_IS_CLEAR",
    13: "WHILE_NOT_FRONT_IS_CLEAR",
    # IFs
    14: "IF_MARKERS_PRESENT",
    15: "IF_NO_MARKERS_PRESENT",
    16: "IF_LEFT_IS_CLEAR",
    17: "IF_RIGHT_IS_CLEAR",
    18: "IF_FRONT_IS_CLEAR",
    19: "IF_NOT_LEFT_IS_CLEAR",
    20: "IF_NOT_RIGHT_IS_CLEAR",
    21: "IF_NOT_FRONT_IS_CLEAR",
    # IFElSEs
    22: "IFELSE_MARKERS_PRESENT",
    23: "IFELSE_NO_MARKERS_PRESENT",
    24: "IFELSE_LEFT_IS_CLEAR",
    25: "IFELSE_RIGHT_IS_CLEAR",
    26: "IFELSE_FRONT_IS_CLEAR",
    27: "IFELSE_NOT_LEFT_IS_CLEAR",
    28: "IFELSE_NOT_RIGHT_IS_CLEAR",
    29: "IFELSE_NOT_FRONT_IS_CLEAR",
    # REPEATs
    30: "REPEAT_1",
    31: "REPEAT_2",
    32: "REPEAT_3",
    33: "REPEAT_4",
    34: "REPEAT_5",
    35: "REPEAT_6",
    36: "REPEAT_7",
    37: "REPEAT_8",
    38: "REPEAT_9",
    39: "REPEAT_10",
    40: "REPEAT_11",
    41: "REPEAT_12",
    # END/PAD/START
    42: "END",
    43: "PAD",
    44: "START",
    # Should not pe produced by the model -------------------------------
    # CURSOR/IFBODY/ELSEBODY/REPEATBODY/WHILEBODY
    45: "CURSOR",
    46: "IFBODY",
    47: "ELSEBODY",
    48: "REPEATBODY",
    49: "WHILEBODY"
}

tok2idx = {v: k for k, v in idx2tok.items()}

vocab_size = len(idx2tok) - nb_helper_tkns
