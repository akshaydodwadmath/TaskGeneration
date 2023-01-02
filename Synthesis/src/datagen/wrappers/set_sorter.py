import json

from tqdm import tqdm

from src.karel_codetask_scoring.finalscore import compute_evaluation_score
from src.karel_emulator.code import Code
from src.karel_emulator.task import Task


def sort_set(all_codes_path):
    """
    Sort the set of all codes by their score
    """
    all_codes = []
    with open(all_codes_path, 'r') as f:
        for codetask in tqdm(f):
            json_task = json.loads(codetask)
            task = Task.parse_json(json_task)
            code = Code.parse_json(json_task['solution'])

            score, info = compute_evaluation_score(code, task, Task([], [], task.type))
            json_task['score'] = score
            json_task['score_info'] = info

            all_codes.append(json_task)

    all_codes.sort(key=lambda x: x['score'], reverse=True)

    with open(f'{all_codes_path.split(".json")[0]}_sorted.json', 'w') as f:
        for codetask in all_codes:
            f.write(json.dumps(codetask))
            f.write('\n')


if __name__ == '__main__':
    sort_set('../../../datasets/synthetic/karelgym_10k/combined_set_456.json')
