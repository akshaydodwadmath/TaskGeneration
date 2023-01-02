from src.karel_codetask_scoring.finalscore import compute_evaluation_score
from src.karel_emulator.code import Code
from src.karel_emulator.fast_emulator import FastEmulator
from src.karel_emulator.task import Task
from src.karel_symexecution.decision_makers import DecisionMaker
from src.karel_symexecution.post_processor import BlockedPostProcessor, \
    EmptySpacePostProcessor
from src.karel_symexecution.symworld import SymWorld
from src.neural_code2task_syn.data import CodeDataset
from src.neural_code2task_syn.decision_makers import IntelligentDecisionMaker
from src.neural_code2task_syn.training import MAX_WORLD_SIZE


class Callback:
    def execute(self):
        raise NotImplementedError()


class EvaluationCallback(Callback):
    def __init__(self,
                 agent: IntelligentDecisionMaker,
                 val_dataset: CodeDataset):
        self.agent = agent
        self.val_dataset = val_dataset

    def execute(self):
        emulator = FastEmulator(1000, 1000)

        self.agent.eval()
        self.agent.has_buffer = False

        mean_score = 0
        for example in self.val_dataset:
            code = Code.parse_json(example["code"])

            if "ref_task" in example:
                ref_task = Task.parse_json(example["ref_task"])
            else:
                ref_task = Task([], [], code.type)

            if 'rows' in example and 'cols' in example:
                rows = example['rows']
                cols = example['cols']
            else:
                rows, cols = MAX_WORLD_SIZE

            symworld = SymWorld.empty_init(rows, cols, self.agent)
            res = emulator.emulate(code, symworld)

            if code.type == "hoc":
                post_processor = BlockedPostProcessor()
            else:
                post_processor = EmptySpacePostProcessor()

            inp_world, out_world = post_processor.symworld_to_world(
                res.outgrid)

            task = Task([inp_world],
                        [out_world],
                        type_=code.type)

            score, _ = compute_evaluation_score(res, code, task,
                                                ref_task)

            # TODO: maybe we should set the score to 0 if the constraints (REDUNDANCY, etc) are not satisfied?

            mean_score += score

        mean_score /= len(self.val_dataset)
        self.agent.has_buffer = True

        return mean_score
