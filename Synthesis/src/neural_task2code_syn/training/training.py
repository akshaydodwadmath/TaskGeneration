import numpy as np
import torch
import wandb
from decouple import config
from torch import autograd
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.neural_task2code_syn.agent.lstm_agent.lstm_agent import LstmAgent
from src.neural_task2code_syn.agent.lstm_agent.network import IOsEncoder, \
    MultiIOProgramDecoder
from src.neural_task2code_syn.karelgym.karel_gym_teacher import KarelGymTeacher
from src.neural_task2code_syn.training.callbacks import EvaluationCallback
from src.neural_task2code_syn.utils.enums import TrainingType
from src.neural_task2code_syn.utils.vocab import tok2idx, vocab_size

# TODO do argparse
API_KEY = config("API_KEY")


def train(agt, env, criter, batch_sz, n_epochs, optimizer,
          training_type: TrainingType, end_of_batch_callbacks=None, scheduler=None):
    """
    Function to train agent and log train/val metrics with W&B
    """

    agt.set_train()
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
            optimizer.zero_grad()

            if training_type == TrainingType.SUPERVISED:
                loss = agt.compute_gradients(training_type, batch, env, criter)
                wandb.log(
                    {"supervised_loss": loss, "nb_batch": nb_batch, "epoch": epoch})
                wandb.log({"batch_idx": batch_idx, "epoch": epoch})
                loss.backward()
            elif training_type == TrainingType.RL:
                reward, variables, grad_variables = agt.compute_gradients(training_type, batch,
                                                                          env, env.nb_rollouts)
                wandb.log({"rl_reward": reward, "nb_batch": nb_batch, "epoch": epoch})
                autograd.backward(variables, grad_variables)
            elif training_type == TrainingType.BEAM_RL:
                reward, variables = agt.compute_gradients(training_type, batch, env,
                                                          env.nb_rollouts, mini_batch_size=8)
                for variable in variables:
                    variable.backward()
                wandb.log({"beam_rl_reward": reward, "nb_batch": nb_batch})

            optimizer.step()

        if end_of_batch_callbacks is not None:
            metric = None
            for callback in end_of_batch_callbacks:
                out = callback.execute(epoch, training_type)
                if metric is None and out is not None:
                    metric = out
            if metric is not None and scheduler is not None:
                scheduler.step(metric)


def sweep_main():
    kernel_size = 3
    hidden_size = 256
    nb_layers = 1
    fc_stack = [512]
    conv_stack = [64, 64, 64]
    embedding_dim = 128
    nb_rollouts = 100
    top_k = [1, 5, 10]
    vocab = tok2idx
    device = 'cuda'
    supervised_batch_size = 32
    rl_batch_size = 16
    n_epochs = 200
    rl_n_epochs = 50
    supervised_learning_rate = 0.0005
    rl_learning_rate = 0.00008
    val_freq = 1
    dataset_type = "karelgym"

    if dataset_type == "full":
        data_pt = "iclr18_data_in_karelgym_format_1m"

    elif dataset_type == "medium":
        set = 2
        data_version = f"set{set}"
        data_pt = f"iclr18_data_in_karelgym_format_100k/{data_version}"

    elif dataset_type == "karelgym":
        data_pt = f"karelgym_10k"

    else:
        set = 3
        data_version = f"set{set}"
        data_pt = f"iclr18_data_in_karelgym_format_10k/{data_version}"

    traindata_path = f"/AIML/misc/work/vpadurea/karel-rl-benchmarks_code_adishs-github/datasets/synthetic/{data_pt}/train.json"
    evaldata_path = f"/AIML/misc/work/vpadurea/karel-rl-benchmarks_code_adishs-github/datasets/synthetic/iclr18_data_in_karelgym_format_1m/val.json"

    config = {
        'supervised_learning_rate': supervised_learning_rate,
        'rl_learning_rate': rl_learning_rate,
        'supervised_batch_size': supervised_batch_size,
        'rl_batch_size': rl_batch_size,
        'kernel_size': kernel_size,
        'conv_stack': conv_stack,
        'fc_stack': fc_stack,
        'tgt_vocabulary_size': vocab_size,
        'tgt_embedding_dim': embedding_dim,
        'decoder_lstm_hidden_size': hidden_size,
        'decoder_nb_lstm_layers': nb_layers,
        'nb_rollouts': nb_rollouts,
        'top_k': top_k,
        'n_epochs': n_epochs,
    }

    wandb.login(key=API_KEY)
    wandb.init(project="karel-rl-benchmark", entity="machine_teaching",
               name=f"supervised_lstm_scheduler_{dataset_type}/hidde"
                    f"n{hidden_size}_embed"
                    f"{embedding_dim}_lr{supervised_learning_rate}_bs"
                    f"{supervised_batch_size}",
               group="LSTM_Agent",
               config=config,
               mode="disabled")

    config = wandb.config

    training_env = KarelGymTeacher(nb_rollouts=nb_rollouts, data_path=traindata_path,
                                   device=device)
    test_env = KarelGymTeacher(nb_rollouts=nb_rollouts, data_path=evaldata_path,
                               device=device)

    agent = LstmAgent(encoder=IOsEncoder(config.kernel_size,
                                         config.conv_stack,
                                         config.fc_stack),
                      decoder=MultiIOProgramDecoder(vocab_size,
                                                    config.tgt_embedding_dim,
                                                    config.fc_stack[-1],
                                                    config.decoder_lstm_hidden_size,
                                                    config.decoder_nb_lstm_layers),
                      vocab=vocab,
                      device=device)

    # Top-k needs to be a list
    eval_callback = EvaluationCallback(agent, test_env, nb_rollouts=nb_rollouts,
                                       top_k=top_k, save=True,
                                       every_n_epoch=val_freq)

    weight_mask = torch.ones(vocab_size).to(device)
    weight_mask[tok2idx['PAD']] = 0
    criterion = nn.CrossEntropyLoss(weight=weight_mask)

    # TODO add option to for type of training

    optimizer = optim.Adam(agent.get_parameters(), lr=config.supervised_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max', threshold=1, patience=2,
                                  factor=0.2, threshold_mode='abs')

    # Train Supervised
    train(agent, training_env, criterion,
          config.supervised_batch_size,
          n_epochs=config.n_epochs,
          training_type=TrainingType.SUPERVISED,
          optimizer=optimizer,
          end_of_batch_callbacks=[eval_callback])

    # After supervised training the best model is picked.
    # best_models_path = "/AIML/misc/work/gtzannet/karel-rl-benchmarks_code_adishs" \
    #                    "-github/src/models/1ypjgcl1/"
    #
    # agent.load_model(best_models_path)
    #
    # # Train RL
    # train(agent, training_env, criterion, rl_batch_size,
    #       n_epochs=rl_n_epochs,
    #       training_type=TrainingType.RL,
    #       optimizer=torch.optim.Adam(agent.get_parameters(), lr=rl_learning_rate),
    #       end_of_batch_callbacks=[eval_callback])


def normal_main():
    kernel_size = 3
    hidden_size = 256
    nb_layers = 1
    fc_stack = [512]
    conv_stack = [64, 64, 64]
    embedding_dim = 128
    nb_rollouts = 100
    top_k = [1, 5, 10]
    vocab = tok2idx
    device = 'cuda'
    supervised_batch_size = 32
    rl_batch_size = 16
    n_epochs = 200
    rl_n_epochs = 50
    supervised_learning_rate = 0.0005
    rl_learning_rate = 0.00008
    val_freq = 1
    dataset_type = "small"

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
        'supervised_learning_rate': supervised_learning_rate,
        'rl_learning_rate': rl_learning_rate,
        'supervised_batch_size': supervised_batch_size,
        'rl_batch_size': rl_batch_size,
        'kernel_size': kernel_size,
        'conv_stack': conv_stack,
        'fc_stack': fc_stack,
        'tgt_vocabulary_size': vocab_size,
        'tgt_embedding_dim': embedding_dim,
        'decoder_lstm_hidden_size': hidden_size,
        'decoder_nb_lstm_layers': nb_layers,
        'nb_rollouts': nb_rollouts,
        'top_k': top_k,
    }

    wandb.login(key=API_KEY)
    wandb.init(project="karel-rl-benchmark", entity="machine_teaching",
               name=f"supervised_lstm_scheduler_{dataset_type}/hidde"
                    f"n{hidden_size}_embed"
                    f"{embedding_dim}_lr{supervised_learning_rate}_bs"
                    f"{supervised_batch_size}",
               group="LSTM_Agent",
               config=config,
               mode="disabled")

    training_env = KarelGymTeacher(nb_rollouts=nb_rollouts, data_path=traindata_path,
                                   device=device)
    test_env = KarelGymTeacher(nb_rollouts=nb_rollouts, data_path=evaldata_path,
                               device=device)

    agent = LstmAgent(encoder=IOsEncoder(kernel_size, conv_stack, fc_stack),
                      decoder=MultiIOProgramDecoder(vocab_size,
                                                    embedding_dim,
                                                    fc_stack[-1],
                                                    hidden_size,
                                                    nb_layers),
                      vocab=vocab,
                      device=device)

    # Top-k needs to be a list
    eval_callback = EvaluationCallback(agent, test_env, nb_rollouts=nb_rollouts,
                                       top_k=top_k, save=True,
                                       every_n_epoch=val_freq)

    weight_mask = torch.ones(vocab_size).to(device)
    weight_mask[tok2idx['PAD']] = 0
    criterion = nn.CrossEntropyLoss(weight=weight_mask)

    # TODO add option to for type of training

    optimizer = optim.Adam(agent.get_parameters(), lr=supervised_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max', threshold=1, patience=2,
                                  factor=0.2, threshold_mode='abs')

    # Train Supervised
    train(agent, training_env, criterion, supervised_batch_size,
          n_epochs=n_epochs,
          training_type=TrainingType.SUPERVISED,
          optimizer=optimizer,
          end_of_batch_callbacks=[eval_callback],
          scheduler=scheduler)

    # After supervised training the best model is picked.
    # best_models_path = "/AIML/misc/work/gtzannet/karel-rl-benchmarks_code_adishs" \
    #                    "-github/src/models/1ypjgcl1/"
    #
    # agent.load_model(best_models_path)
    #
    # # Train RL
    # train(agent, training_env, criterion, rl_batch_size,
    #       n_epochs=rl_n_epochs,
    #       training_type=TrainingType.RL,
    #       optimizer=torch.optim.Adam(agent.get_parameters(), lr=rl_learning_rate),
    #       end_of_batch_callbacks=[eval_callback])


def main():
    sweep_main()
