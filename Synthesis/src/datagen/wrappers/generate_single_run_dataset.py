# TODO postprocess dataset for statistics of the grids/code. See paper for interesting features
# Generate a dataset in a single run
import json
import random
import time

import numpy as np
from tqdm import tqdm

from config_one_taskcode import CODE_TYPES, GLOBAL_WHILECOUNTER_MIN, \
    GLOBAL_WHILECOUNTER_MAX, GLOBAL_MINROWS, GLOBAL_MAXROWS, GLOBAL_MINCOLS, \
    GLOBAL_MAXCOLS, \
    MAXITERS_TASKGRID, MAX_NUM_BLOCKS, DATASET_SIZE, DIST_CODETYPE
from generate_one_taskcode import generate_taskcode

WARMUP_TRIES = 1000


def postprocess_dataset(data_filepath):
    """
    Hold distribution of salient variables
    Grid, Marker ratio, wall ratio, marker count, number of grids
    Program size(number of tokens), control flow ratio, nested control flow # TODO Maybe number of constructs per tasks
    """

    grid_size = []
    marker_rt = []
    wall_rt = []
    marker_count = []
    nb_grids = []
    prg_size = []
    cf_rt = []
    nested_cf_rt = []

    # Postprocessing to return salient variables of full dataset
    with open(f"{data_filepath}/train.json", 'r') as dataset:
        for line in dataset.readlines():
            sample = json.loads(line)
            for example in sample["examples"]:
                row = example["inpgrid_json"]["rows"]
                col = example["inpgrid_json"]["cols"]
                grid_size.append([row, col])
                walls = list(example["inpgrid_json"]["blocked"].split(" "))
                markers = list(example["inpgrid_json"]["markers"].split(" "))

                if walls[0] == str(''):
                    wall_rt.append(0)
                else:
                    wall_rt.append(round(len(walls) / (row * col), 2))

                marker_rt.append(round(len(markers) / (row * col), 2))

                # Store the marker count. Mean marker count per input grid
                num_markers = 0
                for marker in markers:
                    if marker == str(''):
                        pass
                    else:
                        lst_marker = list(marker.split(":"))
                        num_markers += int(lst_marker[-1])

                marker_count.append(round(num_markers / len(markers), 2))

            nb_grids.append(sample["num_examples"])
            max_blocks_allowed = sample["num_blocks_allowed"]
            prg_size.append(max_blocks_allowed)
            # Convert string type_blocks_allowed to list of tokens
            lst_type_blocks = list(sample["type_blocks_allowed"].split(","))
            cf_tkn = ["ifelse", "if", "repeat", "while"]
            cf_count = 0
            for token in lst_type_blocks:
                if token in cf_tkn:
                    cf_count += 1
            cf_rt.append(round(cf_count / max_blocks_allowed, 2))

        sal_vars_dict = {"grid_size": grid_size,
                         "marker_rt": marker_rt,
                         "wall_rt": wall_rt,
                         "marker_count": marker_count,
                         "nb_grids": nb_grids,
                         "prg_size": prg_size,
                         "cf_rt": cf_rt,
                         "nested_cf_rt": nested_cf_rt}

        # Store metadata of the dataset
        metadata_filepath = f"{data_filepath}/metadata.json"
        f_metadata = open(metadata_filepath, 'w', encoding="utf-8")
        f_metadata.write(json.dumps(sal_vars_dict, ensure_ascii=False))
    return


def gen_dataset(nb_tries, data_filepath, postprcs=False):
    """
    Generate dataset on a single run
    """
    st = time.time()
    all_codes_set = set()
    # Used to maintain current distribution
    current_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    f_data = open(f"{data_filepath}", 'w', encoding="utf-8")
    for i in tqdm(range(nb_tries)):

        # For half of tries select round-robin on the code type and for the remaining pick the one that has the furthest
        # distribution from the target
        if i < WARMUP_TRIES:  # Fewer iteration lead to more control in the distribution
            c_type_id = i % len(CODE_TYPES)
        else:
            c_type_id = np.argmax(np.fromiter(DIST_CODETYPE.values(), dtype=float)
                                  - np.fromiter(current_dist.values(), dtype=float))

        # Each code type has a minimum number of blocks
        code_type, min_block_type = CODE_TYPES[c_type_id]

        num_blocks = random.randint(min_block_type, MAX_NUM_BLOCKS)

        # One code pair
        out_json = generate_taskcode(code_type, num_blocks=num_blocks,
                                     whilecounter_min=GLOBAL_WHILECOUNTER_MIN,
                                     whilecounter_max=GLOBAL_WHILECOUNTER_MAX,
                                     minrows=GLOBAL_MINROWS, maxrows=GLOBAL_MAXROWS,
                                     mincols=GLOBAL_MINCOLS,
                                     maxcols=GLOBAL_MAXCOLS,
                                     maxiters=MAXITERS_TASKGRID)
        # Output json is a task code pair that is already checked
        if out_json is not None:  # No valid code produced

            # Use hashmap for duplicates
            if json.dumps(out_json["solution"]["program_json"]) not in all_codes_set:
                f_data.write(json.dumps(out_json, ensure_ascii=False))
                f_data.write('\n')
                all_codes_set.add(json.dumps(out_json["solution"]["program_json"]))
                # Update the current distribution
                current_dist[c_type_id] += 1 / DATASET_SIZE
            else:
                pass

        # Target size of dataset is reached
        if len(all_codes_set) == DATASET_SIZE:
            print("Dataset complete")
            break
    f_data.close()
    print(len(all_codes_set))
    print("number of tries", i)
    print(current_dist)
    elapsed_time = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Generate a metadata json for interesting variables of the dataset
    if postprcs:
        postprocess_dataset(data_filepath)


if __name__ == '__main__':
    gen_dataset(nb_tries=10_000,
                data_filepath="../../../datasets/synthetic/karelgym_10k/try_-3.json")
