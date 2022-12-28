import json
import random

from tqdm import tqdm

"""
Create train/test sets from the original ICLR18' test.json which consists only one I/O pairs and 2 I/O pairs
"""

testdata_path = "../../datasets/synthetic/iclr18_data_in_karelgym_format_1m/test.json"
numbers_of_IOs = 1
f_data = open(
    f"../../datasets/synthetic/iclr18_data_in_karelgym_format_1m/test{numbers_of_IOs}IO.json",
    'w', encoding="utf-8")
#  Load test set and randomly select one and/or two IO pairs and save this version of the set.
with open(testdata_path) as f:
    for codetask in tqdm(f):
        json_task = json.loads(codetask)
        examples = json_task["examples"]
        nums = random.sample(range(0, len(examples) - 1), numbers_of_IOs)
        pregrids = [example['inpgrid_json'] for example in examples]
        postgrids = [example['outgrid_json'] for example in examples]
        new_pregrids = [pregrids[i] for i in nums]
        new_postgrids = [postgrids[i] for i in nums]

        json_task["num_examples"] = numbers_of_IOs
        json_task["examples"] = [{"example_index": i, "inpgrid_json": new_pregrids[i],
                                  "outgrid_json": new_postgrids[i]}
                                 for i in range(len(new_pregrids))]
        f_data.write(json.dumps(json_task, ensure_ascii=False))
        f_data.write('\n')

    f_data.close()
