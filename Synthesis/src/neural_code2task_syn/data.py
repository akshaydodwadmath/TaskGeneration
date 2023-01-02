import json
import os
from pprint import pprint

from torch.utils.data import Dataset
from tqdm import tqdm

from src.karel_emulator.code import Code
from src.karel_emulator.fast_emulator import FastEmulator
from src.karel_emulator.task import Task
from src.karel_symexecution.post_processor import CopyPostProcessor
from src.karel_symexecution.symworld import SymWorld
from src.neural_code2task_syn.decision_makers import ImitatingDecisionMaker


def create_imitation_dataset(tasks_filepath, saves_filepath):
    """Create imitation learning dataset from a tasks.json file."""

    if os.path.isfile(f"{saves_filepath}/imitation_dataset.json"):
        print("Imitation dataset already exists. Delete it to recreate.")
        return

    post_processor = CopyPostProcessor()

    with open(tasks_filepath, 'r') as dataset:
        i = 0
        for line in tqdm(dataset.readlines()):
            i += 1
            task_dict = json.loads(line)

            task = Task.parse_json(task_dict)
            code = Code.parse_json(task_dict["solution"])

            emulator = FastEmulator(1000, 1000)

            if i == 9:
                print(i)

            for pregrid, postgrid in zip(task.pregrids, task.postgrids):
                decision_maker = ImitatingDecisionMaker(emulator=emulator,
                                                        pregrid=pregrid)

                symworld = SymWorld.empty_init(pregrid.rows, pregrid.cols,
                                               decision_maker=decision_maker)

                res = emulator.emulate(code, symworld)

                pre_res, post_res = post_processor.symworld_to_world(res.outgrid,
                                                                     pregrid, postgrid)

                assert pre_res == pregrid
                assert post_res == postgrid

                decision_maker.save_buffer(f"{saves_filepath}/imitation_dataset.json",
                                           code)
                decision_maker.reset_buffer()


class CodeDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        self.data = []
        with open(file_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    # create_imitation_dataset("../../datasets/synthetic/karelgym_10k/try_-3.json",
    #                          "../../datasets/synthetic/karelgym_10k/")

    for line in open(
            "../../datasets/synthetic/karelgym_10k/combined_set_456_sorted.json", 'r'):
        task_dict = json.loads(line)
        task = Task.parse_json(task_dict)
        if task_dict['score_info']['redundancy'] == 'NOT RESPECTED':
            continue
        for pre, post in zip(task.pregrids, task.postgrids):
            pprint(Code.parse_json(task_dict["solution"]).astJson)
            pprint(task_dict["score_info"])
            print(pre.draw())
            print(post.draw())
            print()
        input()
