import copy

import numpy as np
import wandb
from decouple import config
from src.agent.debugger_agent.batch_processor import KarelLGRLRefineBatchProcessor
# from src.agent.debugger.debugger import KarelLGRLRefineModel
from src.agent.debugger_agent.debugger_agent import KarelLGRLRefineModel
from src.karelgym.karel_gym_teacher import KarelGymTeacher
from src.training.callbacks import EvaluationCallback
from src.utils.mutation import mutate_n
from src.utils.vocab import tok2idx
from tqdm import tqdm

if __name__ == '__main__':
    API_KEY = config("API_KEY")

    n_epochs = 50
    batch_sz = 32
    vocab = tok2idx
    set = 3
    data_version = f"set{set}"
    dataset_type = "full"
    device = 'cuda'

    karel_hidden_size = 128
    learning_rate = 0.1
    gradient_clip = 1
    lr_decay_steps = 100000
    lr_decay_rate = 0.5
    max_beam_trees = 5
    max_decoder_length = 100
    use_length_penalty = False
    length_penalty_factor = 0.7
    nb_rollouts = 100
    top_k_eval = 1

    if dataset_type == "full":
        data_pt = "iclr18_data_in_karelgym_format_1m"
    elif dataset_type == "medium":
        set = 2
        data_version = f"set{set}"
        data_pt = f"iclr18_data_in_karelgym_format_100k/{data_version}"
    else:
        set = 3
        data_version = f"set{set}"
        data_pt = f"iclr18_data_in_karelgym_format_10k/{data_version}"

    traindata_path = f"/AIML/misc/work/vpadurea/karel-rl-benchmarks_code_adishs-github/datasets/synthetic/{data_pt}/train.json"
    evaldata_path = f"/AIML/misc/work/vpadurea/karel-rl-benchmarks_code_adishs-github/datasets/synthetic/iclr18_data_in_karelgym_format_1m/val.json"

    config = {
        'karel_hidden_size': karel_hidden_size,
        'learning_rate': learning_rate,
        'gradient_clip': gradient_clip,
        'lr_decay_steps': lr_decay_steps,
        'lr_decay_rate': lr_decay_rate,
        'max_beam_trees': max_beam_trees,
        'max_decoder_length': max_decoder_length,
        'use_length_penalty': use_length_penalty,
        'length_penalty_factor': length_penalty_factor,
        'nb_rollouts': nb_rollouts,
        'top_k_eval': top_k_eval,
        'batch_sz': batch_sz,
        'n_epochs': n_epochs,
        'mutations': 5
    }

    wandb.login(key=API_KEY)
    wandb.init(project="karel-rl-benchmark", entity="machine_teaching",
               name=f"debugger-supervised-1m",
               group="Debugger_Agent",
               config=config,
               mode="online")

    m = KarelLGRLRefineModel(
        cuda=True if device == 'cuda' else False,
        vocab=vocab,
        karel_hidden_size=karel_hidden_size,
        learning_rate=learning_rate,
        gradient_clip=gradient_clip,
        lr_decay_steps=lr_decay_steps,
        lr_decay_rate=lr_decay_rate,
        max_beam_trees=max_beam_trees,
        max_decoder_length=max_decoder_length,
        use_length_penalty=use_length_penalty,
        length_penalty_factor=length_penalty_factor,
    )
    m.model.train()

    env = KarelGymTeacher(nb_rollouts=nb_rollouts, data_path=traindata_path,
                          device=device)

    val_env = KarelGymTeacher(nb_rollouts=nb_rollouts, data_path=evaldata_path,
                              device=device)

    dev_batch_processor = KarelLGRLRefineBatchProcessor(vocab=vocab,
                                                        args=None,
                                                        for_eval=True)

    eval_callback = EvaluationCallback(m, val_env, nb_rollouts=nb_rollouts,
                                       top_k=top_k_eval, save=True,
                                       every_n_epoch=1,
                                       batch_preprocessor=dev_batch_processor)

    end_of_batch_callbacks = [eval_callback]
    training_type = 'train'
    scheduler = None

    n_dist = [1, 2, 3, 4, 5]
    n_dist = n_dist / np.sum(n_dist)
    rng = np.random.RandomState()

    # optimizer = optim.Adam(model.get_parameters(), lr=config.supervised_learning_rate)

    ids = np.arange(0, len(env.TASKS))
    nb_batch = 0
    for epoch in tqdm(range(n_epochs)):
        np.random.shuffle(ids)
        batch_idx = 0
        for idx in tqdm(range(0, len(env.TASKS), batch_sz)):
            nb_batch += 1
            batch_idx += 1
            batch_ids = ids[idx:idx + batch_sz]
            batch = env.batch_reset(batch_ids=batch_ids)
            tgt_seq = env.get_expert_trajectories([entry['id'] for entry in batch])

            mutated_seq = [mutate_n(copy.deepcopy(seq),
                                    rng.choice(len(n_dist), p=n_dist) + 1)
                           for seq in tgt_seq]

            assert mutated_seq != tgt_seq

            batch_processor = KarelLGRLRefineBatchProcessor(vocab=vocab, args=None,
                                                            for_eval=False, )
            processed = batch_processor(batch, tgt_seq, mutated_seq)

            res = m.train(processed)

            wandb.log(
                {"supervised_loss": res['loss'], "nb_batch": nb_batch, "epoch": epoch})
            wandb.log({"batch_idx": batch_idx, "epoch": epoch})

        if end_of_batch_callbacks is not None:
            metric = None
            for callback in end_of_batch_callbacks:
                out = callback.execute(epoch, training_type)
                if metric is None and out is not None:
                    metric = out
            if metric is not None and scheduler is not None:
                scheduler.step(metric)

            # print(res)
            #
            # m.model.eval()
            # stats = {'correct': 0, 'total': 0}
            #
            # with torch.no_grad():
            #
            #     nb_correct = {}
            #     nb_gen = {}
            #     nb_sem = {}
            #
            #     for task_id in tqdm(range(len(val_env.TASKS))):
            #         val_batch = val_env.batch_reset(batch_ids=[task_id])
            #         tgt_seq = val_env.get_expert_trajectories([entry['id'] for entry in
            #                                                    val_batch])
            #         dev_batch_processor = KarelLGRLRefineBatchProcessor(vocab=vocab,
            #                                                             args=None,
            #                                                             for_eval=True)
            #         dev_processed = dev_batch_processor(val_batch, tgt_seq, mutated_seq)
            #         # dev_res = m.eval(dev_processed)
            #
            #         top_k_paths, top_k_tgt_seq = m.get_evaluation_data(dev_processed,
            #                                                            tgt_seq,
            #                                                            10,
            #                                                            [5, ])
            #
            #         for i, k_entry in enumerate([5, ]):
            #             paths = top_k_paths[i]
            #             out_tgt_seq = top_k_tgt_seq[i]
            #             exact = (paths == out_tgt_seq).all(dim=2)
            #             all_mask = exact.any(dim=1)
            #             correct = torch.sum(all_mask).item()
            #
            #             if k_entry not in nb_correct:
            #                 nb_correct[k_entry] = []
            #             nb_correct[k_entry].append(correct)
            #
            #             #  Multistep is called with trace and task_id
            #             topk_gen_rewards = []
            #             topk_sem_rewards = []
            #
            #             for bc, top_paths in enumerate(paths):
            #                 for k, path in enumerate(top_paths):
            #
            #                     if exact.data[0][k]:
            #                         topk_gen_rewards.append(1)
            #                         topk_sem_rewards.append(1)
            #                     else:
            #                         _, _, _, info = val_env.multistep(path, task_id)
            #
            #                         if info["true_success"]:
            #                             topk_gen_rewards.append(1)
            #                         else:
            #                             topk_gen_rewards.append(0)
            #
            #                         if info["semantic_success"]:
            #                             topk_sem_rewards.append(1)
            #                         else:
            #                             topk_sem_rewards.append(0)
            #
            #             if any(topk_gen_rewards) == 1:
            #                 batch_gen_rew = 1
            #             else:
            #                 batch_gen_rew = 0
            #
            #             if any(topk_sem_rewards) == 1:
            #                 batch_sem_rew = 1
            #             else:
            #                 batch_sem_rew = 0
            #
            #             if k_entry not in nb_gen:
            #                 nb_gen[k_entry] = []
            #             if k_entry not in nb_sem:
            #                 nb_sem[k_entry] = []
            #
            #             nb_gen[k_entry].append(batch_gen_rew)
            #             nb_sem[k_entry].append(batch_sem_rew)
            # # print(processed)
            #
            # # for i in range(len(batch)):
            #
            # # optimizer.zero_grad()

# TODO: MOCANEALA - the vocabulary of the debugger uses a hack - it's very large so
#  that the mapping of the edit and add operations do not intersect with the end token
#  find (100*) and (100//) in networks.py, batch_processor.py - SOLVED
