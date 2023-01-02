import argparse
import json
import multiprocessing
import os
import random
import time
from multiprocessing import Process, Value

import numpy as np
from tqdm import tqdm

from config_one_taskcode import CODE_TYPES, GLOBAL_WHILECOUNTER_MIN, \
    GLOBAL_WHILECOUNTER_MAX, GLOBAL_MINROWS, GLOBAL_MAXROWS, GLOBAL_MINCOLS, \
    GLOBAL_MAXCOLS, \
    MAXITERS_TASKGRID, MAX_NUM_BLOCKS, DATASET_SIZE, DIST_CODETYPE
from generate_one_taskcode import generate_taskcode

WARMUP_TRIES = 1000


def gen_dataset(shared_queue, run, total_tries, current_dist):
    """
    Generate dataset on a single run
    """
    all_codes_set = set()
    num_tries = 0
    while run.is_set():
        # For half of tries select round-robin on the code type and for the remaining pick the one that has the furthest
        # distribution from the target
        total_tries.value += 1
        num_tries += 1
        if total_tries.value < WARMUP_TRIES:  # Fewer iteration lead to more control in the distribution
            c_type_id = (total_tries.value + os.getpid()) % len(CODE_TYPES)
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
                all_codes_set.add(json.dumps(out_json["solution"]["program_json"]))
                shared_queue.put(json.dumps(out_json))
                # Update the current distribution
                current_dist[c_type_id] += 1 / DATASET_SIZE
            else:
                pass


def master_proc(data_filepath, q, run, total_tries, current_dist):
    st = time.time()
    shared_set = set()
    f_data = open(f"{data_filepath}", 'w', encoding="utf-8")
    pbar = tqdm(total=DATASET_SIZE)
    while True:
        if q.empty() is False:

            item = json.loads(q.get())
            if json.dumps(item["solution"]["program_json"]) not in shared_set:
                f_data.write(json.dumps(item, ensure_ascii=False))
                f_data.write('\n')
                shared_set.add(json.dumps(item))
                pbar.update(1)

            # Target size of dataset is reached
            if len(shared_set) == DATASET_SIZE:
                print("Dataset complete")
                run.clear()
                break
    f_data.close()
    print(len(shared_set))
    print("Total tries", total_tries.value)
    print(current_dist)
    elapsed_time = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)

    args = parser.parse_args()

    data_filepath = f"../../../datasets/synthetic/karelgym_10k/{args.filename}.json"
    num_process = 20
    processes = []
    q = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    run = manager.Event()
    # Used to maintain current distribution
    current_dist = manager.dict({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0})
    run.set()
    total_tries = Value("i", 0)
    p_master = Process(target=master_proc,
                       args=(data_filepath, q, run, total_tries, current_dist))
    for _ in range(num_process):
        p = Process(target=gen_dataset, args=(q, run, total_tries, current_dist))
        p.start()
        processes.append(p)
    p_master.start()
    for p in processes:
        p.join()
    p_master.join()
