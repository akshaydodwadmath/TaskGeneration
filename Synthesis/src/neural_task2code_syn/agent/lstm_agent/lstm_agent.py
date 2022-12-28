"""
Main responsibilities of an agent:
 - have a network (or encoder-decoder)
 - have an environment? [Not really]
 - translate the environment representations to the network representations
 - be able to sample multiple rollouts
 - maybe make the backward pass? this, however, seems more fit for the training
 function
"""
import itertools
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from src.neural_task2code_syn.agent.abstract_agent import AbstractAgent
from src.neural_task2code_syn.agent.lstm_agent.beam import Beam
from src.neural_task2code_syn.agent.lstm_agent.rolls import Rolls
from src.neural_task2code_syn.utils.code_linearizer import traverse_pre_order
from src.neural_task2code_syn.utils.enums import TrainingType


class LstmAgent(AbstractAgent):
    def __init__(self, encoder, decoder, vocab=None, device='cpu', seq_max_len=45):
        super().__init__(encoder, decoder, vocab, device)
        self.max_len = seq_max_len

    def get_parameters(self):
        return chain(self.encoder.parameters(), self.decoder.parameters())

    def set_train(self):
        self.encoder.train()
        self.decoder.train()

    def set_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def load_model(self, best_models_path):
        self.encoder.load_state_dict(torch.load(f"{best_models_path}encoder.pth"))
        self.decoder.load_state_dict(torch.load(f"{best_models_path}decoder.pth"))

    def save_model(self, save_path):
        torch.save(self.encoder.state_dict(), f"{save_path}/encoder.pth")
        torch.save(self.decoder.state_dict(), f"{save_path}/decoder.pth")

    def do_batch(self, type: TrainingType, *args, **kwargs):
        if type == TrainingType.SUPERVISED:
            return self._do_supervised_batch(*args, **kwargs)
        elif type == TrainingType.RL:
            return self._do_rl_batch(*args, **kwargs)
        elif type == TrainingType.BEAM_RL:
            return self._do_beam_rl_batch(*args, **kwargs)
        else:
            raise ValueError("Unknown training type")

    # TODO pass top-k as list
    def beam_sample(self,
                    input_grids, output_grids,
                    max_len,
                    beam_size,
                    top_k):
        io_embeddings = self.encoder(input_grids, output_grids)

        tgt_start = self.vocab['START']
        tgt_end = self.vocab['END']

        batch_size, nb_ios, io_emb_size = io_embeddings.size()
        use_cuda = io_embeddings.is_cuda
        tt = torch.cuda if use_cuda else torch
        force_beamcpu = True

        # TODO take maximum of top-k list
        beams = [
            Beam(beam_size, max(top_k), tgt_start, tgt_end,
                 use_cuda and not force_beamcpu)
            for _ in range(batch_size)]

        lsm = nn.LogSoftmax(dim=1)

        # We will make it a batch size of beam_size
        batch_state = None  # First one is the learned default state
        batch_grammar_state = None
        batch_inputs = Variable(tt.LongTensor(batch_size, 1).fill_(tgt_start),
                                volatile=False)
        batch_list_inputs = [[tgt_start]] * batch_size
        batch_io_embeddings = io_embeddings
        batch_idx = Variable(torch.arange(0, batch_size, 1).long(), volatile=False)
        if use_cuda:
            batch_idx = batch_idx.cuda()
        beams_per_sp = [1 for _ in range(batch_size)]

        for stp in range(max_len):
            # We will just do the forward of one timestep Each of the inputs
            # for a beam appears as a different sample in the batch
            dec_outs, dec_state, \
            batch_grammar_state = self.decoder(batch_inputs,
                                               batch_io_embeddings,
                                               # batch_list_inputs,
                                               batch_state,
                                               batch_grammar_state)
            # dec_outs -> (batch_size*beam_size, 1, nb_out_word)
            # -> the unnormalized/pre-softmax proba for each word
            # dec_state -> 2-tuple (nb_layers, batch_size*beam_size, nb_ios, dim)

            # Get the actual word probability for each beam
            dec_outs = dec_outs.squeeze(1)  # (batch_size*beam_size x nb_out_word)
            lpb_out = lsm(dec_outs)  # (batch_size*beam_size x nb_out_word)

            # Update all the beams, put out of the circulations the ones that
            # have completed
            new_inputs = []
            new_parent_idxs = []
            new_batch_idx = []
            new_beams_per_sp = []
            new_batch_checker = []
            new_batch_list_inputs = []

            sp_from_idx = 0
            lpb_to_use = lpb_out.data
            if force_beamcpu:
                lpb_to_use = lpb_to_use.cpu()
            for i, (beamState, sp_beam_size) in enumerate(zip(beams, beams_per_sp)):
                if beamState.done:
                    new_beams_per_sp.append(0)
                    continue
                sp_lpb = lpb_to_use.narrow(0, sp_from_idx, sp_beam_size)
                is_done = beamState.advance(sp_lpb)
                if is_done:
                    new_beams_per_sp.append(0)
                    sp_from_idx += sp_beam_size
                    continue

                # Get the input for the decoder at the next step
                sp_next_inputs, sp_next_input_list = beamState.get_next_input()
                sp_curr_beam_size = sp_next_inputs.size(0)
                sp_batch_inputs = sp_next_inputs.view(sp_curr_beam_size, 1)
                # Prepare so that for each beam, it's parent state is correct
                sp_parent_idx_among_beam = beamState.get_parent_beams()
                sp_parent_idxs = sp_parent_idx_among_beam + sp_from_idx
                sp_next_batch_idxs = Variable(tt.LongTensor(sp_curr_beam_size).fill_(i),
                                              volatile=False)
                # Get the idxs of the batches
                if use_cuda:
                    sp_batch_inputs = sp_batch_inputs.cuda()
                    sp_parent_idxs = sp_parent_idxs.cuda()
                new_inputs.append(sp_batch_inputs)
                new_beams_per_sp.append(sp_curr_beam_size)
                new_batch_idx.append(sp_next_batch_idxs)
                new_parent_idxs.append(sp_parent_idxs)
                new_batch_list_inputs.extend([[inp] for inp in sp_next_input_list])
                sp_from_idx += sp_beam_size

            assert (sp_from_idx == lpb_to_use.size(
                0))  # have we gone over all the things?
            if len(new_inputs) == 0:
                # All of our beams are done
                break
            batch_inputs = torch.cat(new_inputs, 0)
            batch_idx = torch.cat(new_batch_idx, 0)
            parent_idxs = torch.cat(new_parent_idxs, 0)
            batch_list_inputs = new_batch_list_inputs

            batch_state = (
                dec_state[0].index_select(1, parent_idxs),
                dec_state[1].index_select(1, parent_idxs)
            )
            batch_io_embeddings = io_embeddings.index_select(0, batch_idx)
            beams_per_sp = new_beams_per_sp
            assert (len(beams_per_sp) == len(beams))
        sampled = []
        for k in top_k:
            sampled.append([beam.get_sampled_top_k(k) for beam in beams])
            # for i, beamState in enumerate(beams):
            #     sampled.append(beamState.get_sampled())

        # TODO loop over different top-k's and and create a list of top-k sampled. Use Beam's get_sampled_topk(topk)
        return sampled

    # TODO: copied from bunel's networks
    def sample(self, input_grids, output_grids, max_len, nb_samples):
        io_embeddings = self.encoder(input_grids, output_grids)

        tgt_start = self.vocab['START']
        tgt_end = self.vocab['END']

        if io_embeddings.is_cuda:
            # Depending on which GPU the machine has, it may be faster to do CPU.
            # For the Quadro K420, the CPU is slightly faster
            # On the Tesla K40, the GPU is twice as fast
            tt = torch.cuda
            device = torch.cuda.current_device()
            tt.set_device(device)
        else:
            tt = torch

        # batch_size is going to be a changing thing, it correspond to how many
        # inputs we are passing through the decoder at once. Here, at
        # initialization, it is just the actual batch_size.
        batch_size, nb_ios, io_emb_size = io_embeddings.size()

        # rolls holds the sample output that we are going to collect for each
        # of the outputs

        # Initial proba for what is certainly sampled
        full_proba = Variable(tt.FloatTensor([1]), requires_grad=False, volatile=False)
        rolls = [Rolls(-1, full_proba, nb_samples, -1, device=self.device) for _ in
                 range(batch_size)]

        sm = nn.Softmax(dim=1)

        # Initialising the elements for the decoder
        curr_batch_size = batch_size  # Will vary as we go along in the decoder

        batch_inputs = Variable(tt.LongTensor(batch_size, 1).fill_(tgt_start),
                                volatile=False)
        batch_list_inputs = [[tgt_start]] * batch_size
        # batch_inputs: (curr_batch, ) -> inputs for the decoder step
        batch_state = None  # First one is the learned default state
        batch_grammar_state = None
        batch_io_embeddings = io_embeddings
        # batch_io_embeddings: curr_batch x nb_ios x io_emb_size

        # Info that we will maintain at each timestep, for all of the traces
        # that we are currently expanding. All these list/tensors should have
        # same sizes.
        trajectories = [[] for _ in range(curr_batch_size)]
        # trajectories: List[ List[idx] ] -> trajectory for each trace that we
        #                                    are currently expanding
        multiplicity = [nb_samples for _ in range(curr_batch_size)]
        # multiplicity: List[ int ] -> How many of this trace have we sampled
        cr_list = [roll_idx for roll_idx in range(curr_batch_size)]
        # cr_list: List[ idx ] -> Which roll/sample is it a trace for

        for stp in range(max_len):
            # Do the forward of one time step, for all our traces to expand
            dec_outs, dec_state, \
            batch_grammar_state = self.decoder(batch_inputs,
                                               batch_io_embeddings,
                                               # batch_list_inputs,
                                               batch_state,
                                               batch_grammar_state)
            # dec_outs: curr_batch x 1 x nb_out_word
            # -> the unnormalized/pre-softmax proba for each word
            # dec_state: 2-tuple of nb_layers x curr_batch x nb_ios x dim

            dec_outs = dec_outs.squeeze(1)  # curr_batch x nb_out_word
            pb_out = sm(dec_outs)  # curr_batch x nb_out_word
            to_sample_from = pb_out

            # Prepare the container for what will need to be given to the next
            # steps
            new_trajectories = []
            new_multiplicity = []
            new_cr_list = []
            new_batch_list_inputs = []

            # This needs to be collected for each of the samples we do
            parent = []  # -> idx of the trace of this sampled output
            next_input = []  # -> sampled output
            sp_proba = []  # -> proba of the sampled output

            # Should this be CPU-only?
            # Apparently not, it uses more memory :/
            # to_sample_from = to_sample_from.cpu()
            for trace_idx in range(curr_batch_size):
                new_batch_list_inputs.append([])
                # Iterate over the current trace prefixes that we have
                idx_per_sample = {}  # -> to group the samples that are same

                # We have sampled `multiplicity[trace_idx]` this prefix trace,
                # we try to continue it `multiplicity[trace_idx]` times.
                # This sample is done with replacement.
                choices = torch.multinomial(to_sample_from.data[trace_idx],
                                            multiplicity[trace_idx],
                                            True)
                # choices: (multiplicity, ) -> sampled output

                # We will now join the samples that are identical, to not
                # duplicate their computation
                for sampled in choices:
                    if sampled in idx_per_sample:
                        # We already have this one, just increase its
                        # multiplicity
                        new_multiplicity[idx_per_sample[sampled]] += 1
                    else:
                        # Bookkeeping so that the future ones similar can be
                        # grouped to this one.
                        idx_per_sample[sampled] = len(new_trajectories)

                        # The trajectory that this now creates:prefix + new elt
                        new_traj = trajectories[trace_idx] + [sampled]
                        new_trajectories.append(new_traj)

                        sp_proba.append(pb_out[trace_idx, sampled])

                        # It belongs to the same samples that his prefix
                        # belonged to
                        new_cr_list.append(cr_list[trace_idx])

                        # The prefix for this one was trace_idx in the previous
                        # batch
                        parent.append(trace_idx)

                        # What will need to be fed in the decoder to continue
                        # this new trace created
                        next_input.append(sampled)

                        # This is the first one that we see so it will have a
                        # multiplicity of 1 for now
                        new_multiplicity.append(1)

            # Add these new samples to our book-keeping of all samples
            for traj, multiplicity, cr, sp_pb in zip(new_trajectories,
                                                     new_multiplicity,
                                                     new_cr_list,
                                                     sp_proba):
                rolls[cr].expand_samples(traj, multiplicity, sp_pb)

            to_continue_mask = [inp != tgt_end for inp in next_input]
            # For the next step, drop everything that we don't need to pursue
            # because they reached the end symbol
            curr_batch_size = sum(to_continue_mask)
            if curr_batch_size == 0:
                # There is nothing left to sample from
                break
            # Extract the ones we need to continue
            next_batch_inputs = [inp for inp in next_input if inp != tgt_end]
            batch_inputs = Variable(tt.LongTensor(next_batch_inputs).view(-1, 1),
                                    requires_grad=False, volatile=False)
            batch_list_inputs = [[inp] for inp in next_batch_inputs]
            # Which are the parents that we need to get the state for
            # (potentially multiple times the same parent)
            parents_to_continue = [parent_idx for (parent_idx, to_cont)
                                   in zip(parent, to_continue_mask) if to_cont]
            parent = Variable(tt.LongTensor(parents_to_continue), requires_grad=False,
                              volatile=False)

            # Gather the output for the next step of the decoder
            # parent: curr_batch_size
            batch_state = (
                dec_state[0].index_select(1, parent),
                dec_state[1].index_select(1, parent)
            )
            # batch_state: 2-tuple nb_layers x curr_batch_size x nb_ios x dim
            batch_io_embeddings = batch_io_embeddings.index_select(0, parent)

            # For all the maintained list, keep only the elt to expand
            joint = [(mul, traj, cr) for mul, traj, cr, to_cont
                     in zip(new_multiplicity,
                            new_trajectories,
                            new_cr_list,
                            to_continue_mask)
                     if to_cont]
            multiplicity, trajectories, cr_list = zip(*joint)

        return rolls

    def _do_beam_rl_batch(self, batch, env, beam_size, mini_batch_size):
        inp_grids = [entry['in_grids'] for entry in batch]
        out_grids = [entry['out_grids'] for entry in batch]

        # Stack the list of tensors
        inp_grids = torch.stack(inp_grids, dim=0).to(self.device)
        out_grids = torch.stack(out_grids, dim=0).to(self.device)

        batch_reward = 0
        use_cuda = inp_grids.is_cuda
        tt = torch.cuda if use_cuda else torch
        variables = []
        with torch.no_grad():
            # Get the programs from the beam search
            decoded = self.beam_sample(inp_grids,
                                       out_grids,
                                       self.max_len,
                                       beam_size, [beam_size])[0]

        for start_pos in range(0, len(decoded), mini_batch_size):
            to_score = decoded[start_pos: start_pos + mini_batch_size]
            task_ids = [x['id'] for x in batch[start_pos: start_pos + mini_batch_size]]
            # Eventually add the reference program
            # if rl_use_ref:
            #     references = [target for target in
            #                   targets[start_pos: start_pos + mini_batch_size]]
            #     for ref, candidates_to_score in zip(references, to_score):
            #         for _, predded in candidates_to_score:
            #             if ref == predded:
            #                 break
            #         else:
            #             candidates_to_score.append((None, ref))  # Don't know its lpb

            # Build the inputs to be scored
            nb_cand_per_sp = [len(candidates) for candidates in to_score]
            in_tgt_seqs = []
            preds = [pred for lp, pred in itertools.chain(*to_score)]
            lines = [[self.vocab['START']] + line for line in preds]
            lens = [len(line) for line in lines]
            ib_max_len = max(lens)

            inp_lines = [
                line[:ib_max_len - 1] + [self.vocab['PAD']] * (
                        ib_max_len - len(line[:ib_max_len - 1]) - 1) for line in
                lines
            ]
            out_lines = [
                line[1:] + [self.vocab['PAD']] * (ib_max_len - len(line)) for line in
                lines
            ]
            in_tgt_seq = Variable(torch.LongTensor(inp_lines))
            out_tgt_seq = Variable(torch.LongTensor(out_lines))
            if use_cuda:
                in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()
            out_care_mask = (out_tgt_seq != self.vocab['PAD'])

            inner_batch_in_grids = inp_grids.narrow(0, start_pos, len(to_score))
            inner_batch_out_grids = out_grids.narrow(0, start_pos, len(to_score))

            # Get the scores for the programs we decoded.
            seq_lpb_var = self._score_multiple_decs(inner_batch_in_grids,
                                                    inner_batch_out_grids,
                                                    in_tgt_seq, inp_lines,
                                                    out_tgt_seq, nb_cand_per_sp)
            lpb_var = torch.mul(seq_lpb_var, out_care_mask.float()).sum(1)

            # Compute the reward that were obtained by each of the sampled programs
            per_sp_reward = []
            for task_id, all_decs in zip(task_ids, to_score):
                sp_rewards = []
                for (lpb, dec) in all_decs:
                    sp_rewards.append(env.multistep(torch.IntTensor(dec), task_id)[1])
                per_sp_reward.append(sp_rewards)

            per_sp_lpb = []
            start = 0
            for nb_cand in nb_cand_per_sp:
                per_sp_lpb.append(lpb_var.narrow(0, start, nb_cand))
                start += nb_cand

            # Use the reward combination function to get our loss on the minibatch
            # (See `reinforce.py`, possible choices are RenormExpected and the BagExpected)
            inner_batch_reward = 0
            for pred_lpbs, pred_rewards in zip(per_sp_lpb, per_sp_reward):
                inner_batch_reward += self._reward_comb_function(pred_lpbs,
                                                                 pred_rewards)

            # We put a minus sign here because we want to maximize the reward.
            variables.append(-inner_batch_reward)

            batch_reward += inner_batch_reward.item()

        return batch_reward, variables

    def _reward_comb_function(self, prediction_lpbs, prediction_reward_list):
        '''
        Simplest Reward Combination Function
        Takes as input:
        `prediction_lpbs`: The log probabilities of each sampled programs
        `prediction_reward_list`: The reward associated with each of these
                                  sampled programs.
        Returns the expected reward under the (renormalized so that it sums to 1)
        probability distribution defined by prediction_lbps.
        '''
        # # Method 1:
        # pbs = prediction_lpbs.exp()
        # pb_sum = pbs.sum()
        # pbs = pbs.div(pb_sum.expand_as(pbs))

        # Method 2:
        prediction_pbs = F.softmax(prediction_lpbs, dim=0)

        if prediction_pbs.is_cuda:
            prediction_reward = torch.cuda.FloatTensor(prediction_reward_list)
        else:
            prediction_reward = torch.FloatTensor(prediction_reward_list)
        prediction_reward = Variable(prediction_reward, requires_grad=False)

        return torch.dot(prediction_pbs, prediction_reward)

    def _score_multiple_decs(self, input_grids, output_grids,
                             tgt_inp_sequences, list_inp_sequences,
                             tgt_out_sequences, nb_cand_per_sp):
        '''
        {input,output}_grids: input_batch_size x nb_ios x channels x height x width
        tgt_{inp,out}_sequences: nb_seq_to_score x max_seq_len
        list_inp_sequences: same as tgt_inp_sequences but under list form
        nb_cand_per_sp: Indicate how many sequences each of the row of {input,output}_grids represent
        '''
        assert sum(nb_cand_per_sp) == tgt_inp_sequences.size(0)
        assert len(nb_cand_per_sp) == input_grids.size(0)
        batch_size, seq_len = tgt_inp_sequences.size()
        io_embedding = self.encoder(input_grids, output_grids)

        io_emb_dims = io_embedding.size()[1:]
        expands = [(nb_cands,) + io_emb_dims for nb_cands in nb_cand_per_sp]
        # Reshape the io_embedding to have one per input samples
        all_io_embs = torch.cat([io_embedding.narrow(0, pos, 1).expand(*exp_dim)
                                 for pos, exp_dim in enumerate(expands)], 0)

        dec_outs, _, _ = self.decoder(tgt_inp_sequences,
                                      all_io_embs)

        # We need to get a logsoftmax at each timestep
        dec_outs = dec_outs.contiguous().view(batch_size * seq_len, -1)
        lpb = F.log_softmax(dec_outs, dim=1)
        lpb = lpb.view(batch_size, seq_len, -1)

        out_lpb = torch.gather(lpb, 2, tgt_out_sequences.unsqueeze(2)).squeeze(2)

        return out_lpb

    def _do_rl_batch(self, batch, env, nb_rollouts):
        inp_grids = [entry['in_grids'] for entry in batch]
        out_grids = [entry['out_grids'] for entry in batch]

        # Stack the list of tensors
        inp_grids = torch.stack(inp_grids, dim=0).to(self.device)
        out_grids = torch.stack(out_grids, dim=0).to(self.device)

        rolls = self.sample(inp_grids, out_grids,
                            self.max_len, nb_rollouts)

        for i, roll in enumerate(rolls):
            # Assign the rewards for each sample
            roll.assign_rewards(env, [], batch[i]["id"])

        # Evaluate the performance on the minibatch
        batch_reward = sum(roll.dep_reward for roll in rolls) / len(batch)

        # Get all variables and all gradients from all the rolls
        variables, grad_variables = zip(*self._batch_rolls_reinforce(rolls))

        # Return the value of the loss/reward over the minibatch for convergence
        # monitoring.
        return batch_reward, variables, grad_variables

    def _batch_rolls_reinforce(self, rolls):
        for roll in rolls:
            for var, grad in roll.yield_var_and_grad():
                if grad is None:
                    assert var.requires_grad is False
                else:
                    yield var, grad

    def _do_supervised_batch(self, batch, env, criterion):
        inp_grids = [entry['in_grids'] for entry in batch]
        out_grids = [entry['out_grids'] for entry in batch]

        # Stack the list of tensors
        inp_grids = torch.stack(inp_grids, dim=0).to(self.device)
        out_grids = torch.stack(out_grids, dim=0).to(self.device)

        io_embedding = self.encoder(inp_grids, out_grids)

        tgt_seq = env.get_expert_trajectories([entry['id'] for entry in batch])

        in_tgt_seq, out_tgt_seq, _ = self._get_tgt_seq(tgt_seq)

        decoder_logit, dec_lstm_st, grammar_st = self.decoder(in_tgt_seq, io_embedding)

        nb_predictions = torch.numel(out_tgt_seq.data)

        loss = criterion(
            decoder_logit.contiguous().view(nb_predictions,
                                            decoder_logit.shape[2]),
            out_tgt_seq.view(nb_predictions)
        )

        return loss

    def _translate_grids(self, grids, *args, **kwargs):
        raise NotImplementedError

    def _map_code(self, code):
        return [self.vocab[t] for t in code]

    def _translate_code(self, code):
        traversed_code_tokens = np.array(
            self._map_code(traverse_pre_order(code, teacher_code=True)))
        traversed_code_tokens = np.append(traversed_code_tokens, 0)

        return traversed_code_tokens

    def _get_tgt_seq(self, codes):
        lines = [
            [self.vocab['START']] +
            self._translate_code(code).tolist() +
            [self.vocab['END']]
            for code in codes]

        lens = [len(line) for line in lines]
        max_len = max(lens)

        input_lines = [line[:max_len - 1] + [self.vocab['PAD']] * (
                max_len - len(line[:max_len - 1]) - 1)
                       for line in lines]
        # Drop the first element, should always be the <start> symbol. This makes
        # everything shifted by one compared to the input_lines
        output_lines = [line[1:] + [self.vocab['PAD']] * (max_len - len(line)) for line
                        in lines]

        in_tgt_seq = torch.tensor(input_lines).long().to(self.device)
        out_tgt_seq = torch.tensor(output_lines).long().to(self.device)

        tgt_lines = [line[0:] + [self.vocab['PAD']] * (max_len - len(line)) for line in
                     lines]

        tgt_seq = torch.tensor(tgt_lines).long().to(self.device)

        return in_tgt_seq, out_tgt_seq, tgt_seq

    # TODO option for random sampling and beam sample
    def get_evaluation_data(self, batch, tgt_seq, nb_rollouts=100,
                            top_k=(5,)):  # topk a list

        beam_size = nb_rollouts

        inp_grids = [entry['in_grids'] for entry in batch]
        out_grids = [entry['out_grids'] for entry in batch]

        # Stack the list of tensors
        inp_grids = torch.stack(inp_grids, dim=0).to(self.device)
        out_grids = torch.stack(out_grids, dim=0).to(self.device)

        # rolls = self.sample(inp_grids, out_grids,
        #                     self.max_len, nb_rollouts)

        # paths = []
        # for roll in rolls:
        #     path = heapq.nlargest(top_k, roll.yield_final_trajectories(),
        #                           key=lambda x: x[2])
        #     path = [F.pad(torch.stack(x[0]), (0, self.max_len - len(x[0])),
        #                   value=self.vocab['PAD']) for x in path]
        #     path = torch.stack(path, dim=0)
        #     paths.append(path)

        pre_paths = self.beam_sample(inp_grids, out_grids,
                                     self.max_len, beam_size,
                                     top_k)  # top_k is passed as a list

        # REMEMBER: now prepaths is a list of top-k sampled

        top_k_paths = []
        for pre_path in pre_paths:
            paths = []
            for path in pre_path:
                path = [F.pad(torch.IntTensor(x[1]), (0, self.max_len - len(x[1])),
                              value=self.vocab['PAD']) for x in path]
                path = torch.stack(path, dim=0)
                paths.append(path)

            # paths = paths[0]
            paths = torch.stack(paths, dim=0).to(self.device)

            top_k_paths.append(paths)

        _, out_tgt_seq, _ = self._get_tgt_seq(tgt_seq)

        out_tgt_seq = F.pad(out_tgt_seq, (0, self.max_len - out_tgt_seq.shape[-1]),
                            value=self.vocab['PAD'])
        out_tgt_seq = out_tgt_seq.unsqueeze(1)

        top_k_tgt_seq = []
        for k in top_k:
            out_tgt_seq_dup = out_tgt_seq.repeat(1, k, 1)
            top_k_tgt_seq.append(out_tgt_seq_dup)

        for i, out_tgt_seq in enumerate(top_k_tgt_seq):
            mask = top_k_tgt_seq[i] == self.vocab['PAD']
            top_k_paths[i][mask] = self.vocab['PAD']

        return top_k_paths, top_k_tgt_seq
