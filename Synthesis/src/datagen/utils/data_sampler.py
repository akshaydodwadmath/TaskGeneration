import json
import random

from tqdm import tqdm


def sample_dataset(path, nb_tasks,
                   save_sampled_path="../../datasets/synthetic/iclr18_100k/set3/train.json"):
    tasks = []
    with open(path) as f:
        for example in tqdm(f):
            task = json.loads(example)
            tasks.append(task)
    random.shuffle(tasks)
    if nb_tasks > 0:
        dataset = tasks[:nb_tasks]
    else:
        dataset = tasks
    with open(save_sampled_path, "w") as final:
        for task in dataset:
            final.write(json.dumps(task))
            final.write('\n')


# Example of sampling a subset
if __name__ == '__main__':
    sample_dataset(
        path="/AIML/misc/work/gtzannet/misc/GandRL_for_NPS/data/1m_6ex_karel/train.json",
        nb_tasks=100000,
        save_sampled_path="../../datasets/synthetic/iclr18_100k/set3/trainNEW.json")
