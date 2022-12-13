import copy
import logging
import os.path

import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.neural_task2code_syn.utils.mutation import mutate_n

logging.basicConfig(level=logging.INFO)  # Sets the level for logging info


class Callback:
    def execute(self, *args):
        raise NotImplementedError()


class EvaluationCallback(Callback):
    def __init__(self, agent, env, nb_rollouts, top_k, save, every_n_epoch=1,
                 batch_preprocessor=None):
        self.agent = agent
        self.env = env
        self.nb_rollouts = nb_rollouts
        # Check for topk list
        if isinstance(top_k, list):
            self.top_k = top_k
        else:
            self.top_k = [top_k]
        self.every_n_epoch = every_n_epoch
        self.called_times = 0
        self.best_sem_acc = 0
        self.save = save
        self.save_path = "../models"
        self.batch_preprocessor = batch_preprocessor

    def execute(self, epoch, training_type):

        if self.called_times % self.every_n_epoch != 0:
            return

        self.called_times += 1

        self.agent.set_eval()

        to_return = None

        if self.batch_preprocessor is not None:
            n_dist = [1, 2, 3]
            n_dist = n_dist / np.sum(n_dist)
            rng = np.random.RandomState()

        with torch.no_grad():

            nb_correct = {}
            nb_gen = {}
            nb_sem = {}
            for task_id in tqdm(range(len(self.env.TASKS))):

                batch = self.env.batch_reset(batch_ids=[task_id])
                tgt_seq = self.env.get_expert_trajectories([entry['id'] for entry in
                                                            batch])

                if self.batch_preprocessor is not None:
                    mutated_seq = [mutate_n(copy.deepcopy(seq),
                                            rng.choice(len(n_dist), p=n_dist) + 1)
                                   for seq in tgt_seq]
                    batch = self.batch_preprocessor(batch, tgt_seq, mutated_seq)

                top_k_paths, top_k_tgt_seq = self.agent.get_evaluation_data(batch,
                                                                            tgt_seq,
                                                                            self.nb_rollouts,
                                                                            self.top_k)

                for i, k_entry in enumerate(self.top_k):
                    paths = top_k_paths[i]
                    out_tgt_seq = top_k_tgt_seq[i]
                    exact = (paths == out_tgt_seq).all(dim=2)
                    all_mask = exact.any(dim=1)
                    correct = torch.sum(all_mask).item()

                    if k_entry not in nb_correct:
                        nb_correct[k_entry] = []
                    nb_correct[k_entry].append(correct)

                    #  Multistep is called with trace and task_id
                    topk_gen_rewards = []
                    topk_sem_rewards = []

                    for bc, top_paths in enumerate(paths):
                        for k, path in enumerate(top_paths):

                            if exact.data[0][k]:
                                topk_gen_rewards.append(1)
                                topk_sem_rewards.append(1)
                            else:
                                _, _, _, info = self.env.multistep(path, task_id)

                                if info["true_success"]:
                                    topk_gen_rewards.append(1)
                                else:
                                    topk_gen_rewards.append(0)

                                if info["semantic_success"]:
                                    topk_sem_rewards.append(1)
                                else:
                                    topk_sem_rewards.append(0)

                    if any(topk_gen_rewards) == 1:
                        batch_gen_rew = 1
                    else:
                        batch_gen_rew = 0

                    if any(topk_sem_rewards) == 1:
                        batch_sem_rew = 1
                    else:
                        batch_sem_rew = 0

                    if k_entry not in nb_gen:
                        nb_gen[k_entry] = []
                    if k_entry not in nb_sem:
                        nb_sem[k_entry] = []

                    nb_gen[k_entry].append(batch_gen_rew)
                    nb_sem[k_entry].append(batch_sem_rew)

            for i, k_entry in enumerate(self.top_k):
                total = len(self.env.TASKS)

                all_correct = sum(nb_correct[k_entry])
                logging.info('Top {} exact match: {}/{}'.format(k_entry, all_correct,
                                                                total))
                exact_match = (all_correct / total) * 100
                # Log callback metrics to wandb platform
                wandb.log({
                    f"{training_type}/top{k_entry}_exact_match_percentage": exact_match,
                    "epoch": epoch})

                gen_suc = sum(nb_gen[k_entry])
                logging.info('Top {} generalized success: {}/{}'.format(k_entry,
                                                                        gen_suc, total))
                gen_acc = (gen_suc / total) * 100
                # Log callback metrics to wandb platform
                wandb.log({
                    f"{training_type}/top{k_entry}_generalized_success_percentage": gen_acc,
                    "epoch": epoch})

                sem_suc = sum(nb_sem[k_entry])
                logging.info('Top {} semantic success: {}/{}'.format(k_entry, sem_suc,
                                                                     total))
                sem_acc = (sem_suc / total) * 100
                # Log callback metrics to wandb platform
                wandb.log({
                    f"{training_type}/top{k_entry}_semantic_success_percentage": sem_acc,
                    "epoch": epoch})

                # Save best model based on semantic accuracy. Model is saved as
                # encoder and decoder separately
                if training_type == "eval":
                    to_return = gen_acc
                else:
                    if i == 0:
                        if sem_acc > self.best_sem_acc and self.save:

                            if not os.path.exists(f"{self.save_path}/{wandb.run.id}"):
                                os.makedirs(f"{self.save_path}/{wandb.run.id}")

                            self.agent.save_model(f"{self.save_path}/{wandb.run.id}")
                            self.best_sem_acc = sem_acc

                        to_return = gen_acc
        if training_type == "eval":
            pass
        else:
            self.agent.set_train()

        return to_return
