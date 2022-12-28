from torch import optim, autograd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.karel_codetask_scoring.finalscore import compute_synthesis_score_faster
from src.karel_emulator.code import Code
from src.karel_emulator.fast_emulator import FastEmulator
from src.karel_emulator.task import Task
from src.karel_symexecution.post_processor import BlockedPostProcessor, \
    EmptySpacePostProcessor
from src.karel_symexecution.symworld import SymWorld
from src.neural_code2task_syn.data import CodeDataset
from src.neural_code2task_syn.decision_makers import IntelligentDecisionMaker, \
    get_features_size, get_output_size
from src.neural_code2task_syn.networks import HandmadeFeaturesAndSymworldNetwork

MAX_WORLD_SIZE = (10, 10)


def train(agent: IntelligentDecisionMaker,
          emulator: FastEmulator,
          dataloader: DataLoader,
          optimizer: optim.Optimizer,
          n_epochs: int
          ):
    agent.train()
    for epoch in tqdm(range(n_epochs)):
        for batch_idx, batch in enumerate(dataloader):
            # batch = batch.to(device)

            agent.reset_buffer()
            optimizer.zero_grad()

            for example in batch:
                code = Code.parse_json(example["code"])

                if "ref_task" in example:
                    ref_task = Task.parse_json(example["ref_task"])
                else:
                    ref_task = Task([], [], code.type)

                if 'buffer' in example:
                    agent.set_current_example(example['buffer'])

                if 'rows' in example and 'cols' in example:
                    rows = example['rows']
                    cols = example['cols']
                else:
                    rows, cols = MAX_WORLD_SIZE

                symworld = SymWorld.empty_init(rows, cols, agent)
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

                score, info = compute_synthesis_score_faster(res, code, task,
                                                             ref_task)

                agent.populate_rewards(score)

            batch_rew, variables, gradients = agent.compute_gradients()
            autograd.backward(variables, gradients)

            optimizer.step()


if __name__ == '__main__':
    learning_rate = 10e-5
    kernel_size = 3
    fc_stack = [256]
    conv_stack = [32, 32, 32]
    batch_size = 32
    n_epochs = 10

    emulator = FastEmulator(1000, 1000)

    decision_maker = IntelligentDecisionMaker(
        HandmadeFeaturesAndSymworldNetwork(
            kernel_size=kernel_size,
            conv_stack=conv_stack,
            fc_stack=fc_stack,
            features_size=get_features_size(),
            output_size=get_output_size(),
            img_size=MAX_WORLD_SIZE,
        ),
        emulator=emulator,
        has_buffer=True)

    optimizer = optim.Adam(decision_maker.get_parameters(),
                           lr=learning_rate)

    dataset = CodeDataset("code_collection.json")
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=lambda x: x)

    train(agent=decision_maker,
          dataloader=dataloader,
          emulator=emulator,
          optimizer=optimizer,
          n_epochs=n_epochs)
