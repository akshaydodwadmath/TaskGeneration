"""
Load and evaluate best pretrained model on test dataset
"""
import wandb
from decouple import config

from src.neural_task2code_syn.agent.lstm_agent.lstm_agent import LstmAgent
from src.neural_task2code_syn.agent.lstm_agent.network import IOsEncoder, \
    MultiIOProgramDecoder
from src.neural_task2code_syn.karelgym.karel_gym_teacher import KarelGymTeacher
from src.neural_task2code_syn.training.callbacks import EvaluationCallback
from src.neural_task2code_syn.utils.vocab import tok2idx, vocab_size


def evaluate(end_of_batch_callbacks=None):
    """
    Function to train agent and log train/val metrics with W&B
    """

    if end_of_batch_callbacks is not None:
        for callback in end_of_batch_callbacks:
            callback.execute(1, training_type="eval")


if __name__ == '__main__':
    API_KEY = config("API_KEY")
    kernel_size = 3
    hidden_size = 512
    nb_layers = 1
    fc_stack = [512]
    conv_stack = [64, 64, 64]
    embedding_dim = 128
    nb_rollouts = 32  # It is the beam size
    top_k = [1, 10]
    vocab = tok2idx
    device = 'cuda'

    testdata_path = "../../datasets/realworld/iclr18_data_in_karelgym_format_stanfordkarel/test.json"
    # testdata_path = "../../datasets/synthetic/iclr18_data_in_karelgym_format_1m/test2IO.json"
    # testdata_path = "../../datasets/synthetic/karelgym_10k/test.json"

    best_models_path = "../models/2i9k8ih4/"

    # Alternative best model directory "../models/2i9k8ih4/", "../models/4hj7fjo4/"

    config = {
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
               name=f"NewTableTest2IO",
               group="LSTM_Agent",
               config=config,
               mode="disabled")

    test_env = KarelGymTeacher(nb_rollouts=nb_rollouts, data_path=testdata_path,
                               device=device)

    encoder_model = IOsEncoder(kernel_size, conv_stack, fc_stack)
    decoder_model = MultiIOProgramDecoder(vocab_size, embedding_dim, fc_stack[-1],
                                          hidden_size, nb_layers)

    agent = LstmAgent(encoder=encoder_model,
                      decoder=decoder_model,
                      vocab=vocab,
                      device=device)

    agent.load_model(best_models_path)
    agent.set_eval()

    eval_callback = EvaluationCallback(agent, test_env, nb_rollouts=nb_rollouts,
                                       top_k=top_k, save=False,
                                       every_n_epoch=1)

    # Evaluate trained agent
    evaluate(end_of_batch_callbacks=[eval_callback])
