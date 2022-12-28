from collections import OrderedDict

# code types with min size
CODE_TYPES = [

    # DEPTH = 1
    ({'type': 'run', 'children': [{'type': 'A_1', 'minval': '1'}]}, 1),

    # DEPTH = 2
    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'D_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'}
      ]}, 2),

    # DEPTH = 2
    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'D_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'},
          {'type': 'D_2',
           'children': [
               {'type': 'A_4', 'minval': '1'}
           ]},
          {'type': 'A_5'}
      ]}, 4),

    # DEPTH = 3
    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'D_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'D_2',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'},
           ]},
          {'type': 'A_5'}
      ]}, 3),

    # DEPTH = 3
    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'D_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'D_2',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'}
           ]},
          {'type': 'A_5'},
          {'type': 'D_3',
           'children': [
               {'type': 'A_6', 'minval': '1'}
           ]},
          {'type': 'A_7'}
      ]}, 5),

    # DEPTH = 3
    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'D_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'},
          {'type': 'D_2',
           'children': [
               {'type': 'A_4'},
               {'type': 'D_3',
                'children': [
                    {'type': 'A_5', 'minval': '1'}
                ]},
               {'type': 'A_6'}
           ]},
          {'type': 'A_7'}
      ]}, 5),

    # DEPTH = 3
    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'D_1',
           'children': [
               {'type': 'A_2', 'minval': '1'},
               {'type': 'D_2',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'}
           ]},
          {'type': 'A_5'},
          {'type': 'D_3',
           'children': [
               {'type': 'A_6'},
               {'type': 'D_4',
                'children': [
                    {'type': 'A_7', 'minval': '1'}
                ]},
               {'type': 'A_8'}
           ]},
          {'type': 'A_9'}
      ]}, 6),

    # DEPTH = 4
    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'D_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'D_2',
                'children': [
                    {'type': 'A_3'},
                    {'type': 'D_3',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]},
                    {'type': 'A_5'}
                ]},
               {'type': 'A_6'}
           ]},
          {'type': 'A_7'}
      ]}, 4)

]

# code generation parameters
BASIC_KAREL_ACTION_BLOCKS = ['move', 'turn_left', 'turn_right', 'pick_marker',
                             'put_marker']
BASIC_HOC_ACTION_BLOCKS = ['move', 'turn_left', 'turn_right']

OTHER_KAREL_BLOCKS = [
    'while(bool_path_ahead)', 'while(bool_no_path_ahead)',
    'while(bool_path_left)', 'while(bool_no_path_left)',
    'while(bool_path_right)', 'while(bool_no_path_right)',
    'while(bool_marker_present)', 'while(bool_no_marker_present)',
    'repeat(2)', 'repeat(3)', 'repeat(4)',
    'repeat(5)', 'repeat(6)', 'repeat(7)',
    'repeat(8)', 'repeat(9)', 'repeat(10)',
    'if(bool_path_ahead)', 'if(bool_no_path_ahead)',
    'if(bool_path_left)', 'if(bool_no_path_left)',
    'if(bool_path_right)', 'if(bool_no_path_right)',
    'if(bool_marker_present)', 'if(bool_no_marker_present)',
    'ifelse(bool_path_ahead)', 'ifelse(bool_no_path_ahead)',
    'ifelse(bool_path_left)', 'ifelse(bool_no_path_left)',
    'ifelse(bool_path_right)', 'ifelse(bool_no_path_right)',
    'ifelse(bool_marker_present)', 'ifelse(bool_no_marker_present)'
]

OTHER_HOC_BLOCKS = ['repeat_until_goal(bool_goal)',
                    'repeat(2)', 'repeat(3)', 'repeat(4)',
                    'repeat(5)', 'repeat(6)', 'repeat(7)',
                    'repeat(8)', 'repeat(9)', 'repeat(10)',
                    'if(bool_path_ahead)', 'if(bool_path_left)', 'if(bool_path_right)',
                    'ifelse(bool_path_ahead)', 'ifelse(bool_path_left)',
                    'ifelse(bool_path_right)'
                    ]

MAX_NUM_BLOCKS = 34

# code rollout parameters
GLOBAL_WHILECOUNTER_MIN = 6
GLOBAL_WHILECOUNTER_MAX = 40

# task grid parameters
GLOBAL_MINROWS = 2
GLOBAL_MAXROWS = 16
GLOBAL_MINCOLS = 2
GLOBAL_MAXCOLS = 16
GLOBAL_MAX_MARKERS_IN_CELL = 10
MAXITERS_TASKGRID = 10

DIST_PWALL = OrderedDict({0: 1 / 3, 1: 1 / 3,
                          2: 1 / 3})  # 2 refers to any value between (0,1). Make sure that the probs sum up to 1
DIST_PMARKER = OrderedDict({0: 2 / 3, 1: 1 / 3})  # 1 refers to any value between (0,1]
DIST_MARKERCOUNTS = OrderedDict(
    {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1,
     9: 0.1, 10: 0.1})

# number of task grids
DIST_NUMGRIDS = OrderedDict({1: 0.5, 2: 0.5})
DIST_PROGRAM_TYPE = OrderedDict({"karel": 0.5, "hoc": 0.5})

# Target codetype dist in dataset
DIST_CODETYPE = OrderedDict({0: 0, 1: 1/7, 2: 1/7, 3: 1/7, 4: 1/7, 5: 1/7,
                             6: 1/7, 7: 1/7})
DATASET_SIZE = 10

STORE = "if,ifelse,while,repeat,putMarker,pickMarker"  # Full store. Basic action are always allowed

AVAILABLE_LOCATION_GENERATORS = ["get_all_quadrant_cells", "get_all_cells"]
LOCATIONS_GENERATOR = "get_all_quadrant_cells"