from typing import List

from src.karel_emulator.tokens import blocktypes
from src.karel_emulator.world import World


class Task:
    def __init__(self,
                 pregrids: List[World],
                 postgrids: List[World],
                 type_: str,
                 num_examples: int = -1,
                 num_blocks_allowed: int = 100,
                 type_blocks_allowed: str = blocktypes):

        self.pregrids = pregrids
        self.postgrids = postgrids
        self.type = type_
        self.num_examples = num_examples
        self.num_blocks_allowed = num_blocks_allowed
        self.type_blocks_allowed = type_blocks_allowed

        assert len(self.pregrids) == len(self.postgrids), \
            'pregrids and postgrids must have the same length'

        if num_examples == -1:
            self.num_examples = len(pregrids)

    @classmethod
    def parse_json(cls, task_json):
        type = task_json["task_type"]
        num_examples = int(task_json['num_examples'])
        num_blocks_allowed = int(task_json['num_blocks_allowed'])
        if not isinstance(task_json['type_blocks_allowed'], list):
            type_blocks_allowed = task_json['type_blocks_allowed'].split(",")
        else:
            type_blocks_allowed = task_json['type_blocks_allowed']

        examples = task_json['examples']
        # Create the Task with all IO pairs
        pregrids = [World.parseJson(example['inpgrid_json']) for example in examples]
        postgrids = [World.parseJson(example['outgrid_json']) for example in examples]

        return cls(pregrids, postgrids, type, num_examples,
                   num_blocks_allowed, type_blocks_allowed)

    def pprint(self):
        for i, (preWorld, postWorld) in enumerate(zip(self.pregrids, self.postgrids)):
            pre = preWorld.toString()
            post = postWorld.toString()
            prelines = pre.split("\n")
            postlines = post.split("\n")
            print("\n", f"Grid {i}")
            for j, (preline, postline) in enumerate(zip(prelines, postlines)):
                if j == len(postlines) // 2:
                    print(preline, " --> ", postline)
                else:
                    print(preline, "     ", postline)

    def to_json(self):
        return {
            "task_type": self.type,
            "num_examples": self.num_examples,
            "num_blocks_allowed": self.num_blocks_allowed,
            "type_blocks_allowed": self.type_blocks_allowed,
            "examples": [
                {
                    "inpgrid_json": pregrid.toJson(),
                    "outgrid_json": postgrid.toJson()
                }
                for pregrid, postgrid in zip(self.pregrids, self.postgrids)
            ]
        }


if __name__ == '__main__':
    import json

    with open("../../tests/karel_emulator/test_data/taskany.json", "r") as f:
        taskcode = json.load(f)

    t = Task(taskcode)

    t2 = Task.from_worlds(t.pregrids, t.postgrids,
                          type_=t.type,
                          num_examples=t.num_examples,
                          num_blocks_allowed=t.num_blocks_allowed,
                          type_blocks_allowed=','.join(t.type_blocks_allowed))

    print(t.type)
    print(t.num_examples)
    print(t.num_blocks_allowed)
    print(t.type_blocks_allowed)

    t.pprint()

    t2.pprint()
