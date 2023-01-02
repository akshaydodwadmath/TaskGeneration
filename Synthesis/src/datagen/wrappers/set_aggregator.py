import json
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

from config_one_taskcode import DIST_CODETYPE
from src.datagen.codegen.convert_ast_to_symast import code_to_codetype

"""
Given a list of completed datasets this function will combine them and create a final larger set that satisfies code 
duplicates and codetype distribution by subsampling
"""
eps = 10e-4  # Stopping criterion for matching distributions


def set_aggregator(ls_sets: list, save_path: str):
    all_codes_set = set()
    typelist = []
    full_set = []
    # Initialize current distribution.
    current_num = OrderedDict({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0})
    for karelset in ls_sets:
        with open(karelset) as f:
            for codetask in tqdm(f):
                json_task = json.loads(codetask)
                # Use hashmap for duplicates
                if json.dumps(
                        json_task["solution"]["program_json"]) not in all_codes_set:
                    all_codes_set.add(json.dumps(json_task["solution"]["program_json"]))
                    # Given the ast code define with codetype it is
                    c_type_id = code_to_codetype(json_task["solution"])
                    typelist.append(c_type_id)
                    current_num[c_type_id] += 1
                    full_set.append(json_task)

    # Dataset set so far
    set_length = len(all_codes_set)
    target_dist = np.fromiter(DIST_CODETYPE.values(), dtype=float)
    current_dist = OrderedDict({k: v / set_length for k, v in current_num.items()})
    current_dist = np.fromiter(current_dist.values(), dtype=float)
    while np.linalg.norm(target_dist - current_dist) > eps:
        c_type_id = np.argmin(target_dist - current_dist)
        del_i = typelist.index(c_type_id)
        del typelist[del_i]
        del full_set[del_i]
        # Update distribution
        current_num[int(c_type_id)] -= 1
        current_dist = OrderedDict(
            {k: v / len(full_set) for k, v in current_num.items()})
        current_dist = np.fromiter(current_dist.values(), dtype=float)

    # Write aggregate set to filepath
    f_data = open(f"{save_path}", 'w', encoding="utf-8")
    for codetask in full_set:
        f_data.write(json.dumps(codetask, ensure_ascii=False))
        f_data.write('\n')
    f_data.close()


if __name__ == '__main__':
    # How to use aggregator
    set_aggregator(ls_sets=["../../../datasets/synthetic/karelgym_10k/try_4.json",
                            "../../../datasets/synthetic/karelgym_10k/try_5.json",
                            "../../../datasets/synthetic/karelgym_10k/try_6.json"],
                   save_path="../../../datasets/synthetic/karelgym_10k/combined_set_456"
                             ".json")
