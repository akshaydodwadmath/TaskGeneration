import heapq
import logging

import torch
import torch.nn.functional as F
import wandb
from decouple import config
from src.agent.debugger_agent import beam_search, prepare_spec
from src.agent.debugger_agent.batch_processor import KarelLGRLRefineBatchProcessor
from src.agent.debugger_agent.debugger_agent import maybe_cuda, KarelLGRLRefineModel
from src.agent.lstm_agent.lstm_agent import LstmAgent
from src.agent.lstm_agent.network import IOsEncoder, MultiIOProgramDecoder
from src.karelgym.karel_gym_teacher import KarelGymTeacher
from src.utils.vocab import tok2idx, vocab_size
from tqdm import tqdm


def sed_best_first_search(synthesizer, debugger,
                          env, debugger_batch_preprocessor,
                          synth_rollouts,
                          synth_top_k: int,
                          debugger_rollouts,
                          debugger_top_k: int,
                          allowed_operations=100,
                          max_decoder_length=100,
                          seq_max_len=45,
                          cuda=True):
    synthesizer.set_eval()
    debugger.set_eval()

    if cuda:
        device = "cuda"
    else:
        device = "cpu"

    nb_correct = []
    nb_gen = []
    nb_sem = []
    nb_correct_nodeb = []
    nb_gen_nodeb = []
    nb_sem_nodeb = []
    with torch.no_grad():
        for task_id in tqdm(range(len(env.TASKS))):
            batch = env.batch_reset(batch_ids=[task_id])
            tgt_seq = env.get_expert_trajectories([entry['id'] for entry in
                                                   batch])

            top_k_paths, top_k_tgt_seq = \
                synthesizer.get_evaluation_data(batch,
                                                tgt_seq,
                                                synth_rollouts,
                                                (synth_top_k,))

            batch = debugger_batch_preprocessor(batch, tgt_seq, tgt_seq)  # second
            # tgt_seq is a dummy - we will modify it at each step

            set_of_paths = set()
            set_of_undebugged_paths = set()
            seen_paths = set()
            finished = False
            for bc, top_paths in enumerate(top_k_paths[0]):
                for k, path in enumerate(top_paths):
                    _, _, _, info = env.multistep(path, task_id)
                    if info['solved'] == 1:
                        finished = True
                    set_of_paths.add((info['solved'], tuple(path.tolist())))
                    set_of_undebugged_paths.add((info['solved'], tuple(path.tolist())))

            if not finished:
                for i in range(allowed_operations):

                    path = max(set_of_paths, key=lambda x: x[0])
                    set_of_paths.remove(path)
                    seen_paths.add(path[1])
                    x = torch.IntTensor(path[1])
                    x = [action for action in x.tolist() if action <
                         vocab_size - 3]
                    ref_code = prepare_spec.lists_to_packed_sequence_translated(
                        [x],
                        vocab,
                        False,
                        volatile=False)
                    batch = batch._replace(ref_code=ref_code)

                    input_grids, output_grids, _1, dec_data, ref_code, \
                    ref_trace_grids, ref_trace_events, cag_interleave, _2 = batch
                    if cuda:
                        input_grids = input_grids.cuda(non_blocking=True)
                        output_grids = output_grids.cuda(non_blocking=True)
                        dec_data = maybe_cuda(dec_data, non_blocking=True)
                        ref_code = maybe_cuda(ref_code, non_blocking=True)
                        ref_trace_grids = maybe_cuda(ref_trace_grids, non_blocking=True)
                        ref_trace_events = maybe_cuda(ref_trace_events,
                                                      non_blocking=True)

                    io_embed, ref_code_memory, ref_trace_memory = debugger.model.encode(
                        input_grids, output_grids, ref_code, ref_trace_grids,
                        ref_trace_events, cag_interleave)
                    init_state = debugger.model.decoder.init_state(
                        ref_code_memory, ref_trace_memory,
                        io_embed.shape[0], io_embed.shape[1])
                    memory = debugger.model.decoder.prepare_memory(io_embed,
                                                                   ref_code_memory,
                                                                   ref_trace_memory,
                                                                   ref_code)

                    pre_paths = beam_search.beam_search(len(input_grids),
                                                        init_state,
                                                        memory,
                                                        debugger.model.decode_token,
                                                        debugger_rollouts,
                                                        cuda=cuda,
                                                        max_decoder_length=max_decoder_length,
                                                        return_attention=False,
                                                        return_beam_search_result=False,
                                                        differentiable=False,
                                                        use_length_penalty=False,
                                                        factor=0.7)

                    pre_paths = heapq.nlargest(debugger_top_k, pre_paths[0],
                                               key=lambda x: x[0])
                    pre_paths = [[y[1] for y in pre_paths]]
                    pre_paths = [debugger.model.decoder.postprocess_output(pre_paths,
                                                                           memory)]

                    for pre_path in pre_paths:
                        paths = []
                        for path in pre_path:
                            path = [x + [vocab['END']] for x in path]
                            # path.append(self.vocab['END'])  # TODO: to test exact match
                            path = [F.pad(torch.IntTensor(x), (0, seq_max_len - len(x)),
                                          value=vocab['PAD']) for x in path]
                            path = torch.stack(path, dim=0)
                            paths.append(path)

                        # paths = paths[0]

                        finished = False
                        for bc, top_paths in enumerate(paths):
                            for k, path in enumerate(top_paths):
                                if tuple(path.tolist()) in seen_paths:
                                    continue
                                _, _, _, info = env.multistep(path, task_id)
                                if info['solved'] == 1:
                                    # TODO: evaluate here,
                                    #  we don't need debugging if everything is solved
                                    finished = True
                                set_of_paths.add((info['solved'], tuple(path.tolist())))

                            if finished:
                                break

                    if finished:
                        break

            aux = [torch.IntTensor(x[1]).to(device) for x in
                   set_of_paths]
            aux += [torch.IntTensor(x).to(device) for x in
                    seen_paths]
            paths = torch.stack(aux, dim=0)

            out_tgt_seq = top_k_tgt_seq[0][:, 0, :]
            out_tgt_seq = out_tgt_seq.repeat(1, len(set_of_paths) + len(seen_paths), 1)
            exact = (paths == out_tgt_seq).all(dim=2)
            all_mask = exact.any(dim=1)
            correct = torch.sum(all_mask).item()

            nb_correct.append(correct)

            #  Multistep is called with trace and task_id
            topk_gen_rewards = []
            topk_sem_rewards = []

            for bc, top_paths in enumerate([paths]):
                for k, path in enumerate(top_paths):

                    if exact.data[0][k]:
                        topk_gen_rewards.append(1)
                        topk_sem_rewards.append(1)
                    else:
                        _, _, _, info = env.multistep(path, task_id)

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

            nb_gen.append(batch_gen_rew)
            nb_sem.append(batch_sem_rew)

            paths = torch.stack([torch.IntTensor(x[1]).to(device) for x in
                                 set_of_undebugged_paths],
                                dim=0)

            out_tgt_seq = top_k_tgt_seq[0][:, 0, :]
            out_tgt_seq = out_tgt_seq.repeat(1, len(set_of_undebugged_paths), 1)
            exact = (paths == out_tgt_seq).all(dim=2)
            all_mask = exact.any(dim=1)
            correct = torch.sum(all_mask).item()

            nb_correct_nodeb.append(correct)

            #  Multistep is called with trace and task_id
            topk_gen_rewards = []
            topk_sem_rewards = []

            for bc, top_paths in enumerate([paths]):
                for k, path in enumerate(top_paths):

                    if exact.data[0][k]:
                        topk_gen_rewards.append(1)
                        topk_sem_rewards.append(1)
                    else:
                        _, _, _, info = env.multistep(path, task_id)

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

            nb_gen_nodeb.append(batch_gen_rew)
            nb_sem_nodeb.append(batch_sem_rew)

    total = len(env.TASKS)

    all_correct = sum(nb_correct)
    logging.info('exact match: {}/{}'.format(all_correct,
                                             total))

    gen_suc = sum(nb_gen)
    logging.info('generalized success: {}/{}'.format(gen_suc, total))

    sem_suc = sum(nb_sem)
    logging.info('semantic success: {}/{}'.format(sem_suc,
                                                  total))

    exact_match = (all_correct / total) * 100
    # Log callback metrics to wandb platform
    wandb.log({
        f"eval/exact_match_percentage": exact_match,
        "epoch": 1})
    wandb.log({
        f"eval/generalized_success_percentage": gen_suc / total * 100,
        "epoch": 1})
    wandb.log({
        f"eval/semantic_success_percentage": sem_suc / total * 100,
        "epoch": 1})

    all_correct = sum(nb_correct_nodeb)
    logging.info('exact match nodeb: {}/{}'.format(all_correct,
                                                   total))

    gen_suc = sum(nb_gen_nodeb)
    logging.info('generalized success nodeb: {}/{}'.format(gen_suc, total))

    sem_suc = sum(nb_sem_nodeb)
    logging.info('semantic success nodeb: {}/{}'.format(sem_suc,
                                                        total))

    exact_match = (all_correct / total) * 100
    # Log callback metrics to wandb platform
    wandb.log({
        f"eval/exact_match_percentage_nodeb": exact_match,
        "epoch": 1})
    wandb.log({
        f"eval/generalized_success_percentage_nodeb": gen_suc / total * 100,
        "epoch": 1})
    wandb.log({
        f"eval/semantic_success_percentage_nodeb": sem_suc / total * 100,
        "epoch": 1})


if __name__ == '__main__':
    API_KEY = config("API_KEY")
    kernel_size = 3
    hidden_size = 256
    nb_layers = 1
    fc_stack = [512]
    conv_stack = [64, 64, 64]
    embedding_dim = 128
    nb_rollouts = 32
    vocab = tok2idx
    device = 'cuda'
    top_k_debugger = 32
    top_k_synth = 32

    testdata_path = "/AIML/misc/work/vpadurea/karel-rl-benchmarks_code_adishs-github/datasets/synthetic/iclr18_data_in_karelgym_format_1m/test.json"

    synth_best_models_path = "/AIML/misc/work/gtzannet/karel-rl-benchmarks_code_adishs-github/src/models/1ypjgcl1/"
    debugger_best_models_path = \
        "/AIML/misc/work/vpadurea/karel-rl-benchmarks_code_adishs-github/src/agent" \
        "/models" \
        "/3vvhtkjf"

    config = {
        "beamsize_debugger": nb_rollouts,
        "beamsize_synthetic": nb_rollouts,
        "debugger": "3vvhtkjf",
        "synthesizer": "1ypjgcl1",
        "top_k_debugger": top_k_debugger,
        "top_k_synthesizer": top_k_synth,
        "allowed_operations": 100
    }

    wandb.init(project="karel-rl-benchmark", entity="machine_teaching",
               name=f"DebuggerFullDataTest-best_first",
               group="Debugger_Agent",
               config=config,
               mode="online")

    test_env = KarelGymTeacher(nb_rollouts=nb_rollouts, data_path=testdata_path,
                               device=device)

    encoder_model = IOsEncoder(kernel_size, conv_stack, fc_stack)
    decoder_model = MultiIOProgramDecoder(vocab_size, embedding_dim, fc_stack[-1],
                                          hidden_size, nb_layers)

    agent = LstmAgent(encoder=encoder_model,
                      decoder=decoder_model,
                      vocab=vocab,
                      device=device)

    agent.load_model(synth_best_models_path)

    m = KarelLGRLRefineModel(
        cuda=True,
        vocab=vocab,
        karel_hidden_size=128,
        learning_rate=0.0001,
        gradient_clip=1,
        lr_decay_steps=100000,
        lr_decay_rate=0.5,
        max_beam_trees=5,
        max_decoder_length=100,
        use_length_penalty=False,
        length_penalty_factor=0.7
    )

    m.load_model(debugger_best_models_path)

    dev_batch_processor = KarelLGRLRefineBatchProcessor(vocab=vocab,
                                                        args=None,
                                                        for_eval=True)

    sed_best_first_search(
        agent,
        m,
        test_env,
        dev_batch_processor,
        synth_rollouts=nb_rollouts,
        debugger_rollouts=nb_rollouts,
        synth_top_k=top_k_synth,
        debugger_top_k=top_k_debugger,
    )

# TODO: synth_top_k is different from the max over the set_of_paths, as with one we
#  pick the one that solves the most tasks, but with the other we pick the one that
#  has the highest probability
