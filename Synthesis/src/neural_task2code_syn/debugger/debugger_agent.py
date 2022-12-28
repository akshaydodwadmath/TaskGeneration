import collections
import heapq

import numpy as np
import torch
import torch.nn.functional as F
from src.agent.debugger_agent import beam_search
from src.agent.debugger_agent.networks import LGRLRefineKarel
from src.utils.code_linearizer import traverse_pre_order
from src.utils.vocab import vocab_size
from torch import nn, optim


def code_to_tokens(seq, vocab):
    tokens = []
    for i in seq:
        if i == 1:  # </s>
            break
        tokens.append(vocab.itos(i))
    return tokens


def maybe_cuda(tensor, non_blocking=False):
    if tensor is None:
        return None
    return tensor.cuda(non_blocking=non_blocking)


class Spans(collections.namedtuple('Spans', 'spans')):
    def cuda(self, non_blocking=False):
        return self


class PackedDecoderData(collections.namedtuple('PackedDecoderData', ('input',
                                                                     'output',
                                                                     'io_embed_indices',
                                                                     'ref_code'))):
    def cuda(self, non_blocking=False):
        input_ = maybe_cuda(self.input, non_blocking)
        output = maybe_cuda(self.output, non_blocking)
        io_embed_indices = maybe_cuda(self.io_embed_indices, non_blocking)
        ref_code = maybe_cuda(self.ref_code, non_blocking)
        return PackedDecoderData(input_, output, io_embed_indices, ref_code)


class BaseModel:

    def __init__(self):
        self.last_step = 0
        # self.model_dir = args.model_dir
        # self.save_every_n = args.save_every_n
        # self.debug_every_n = args.debug_every_n
        #
        # # self.saver = saver.Saver(self.model, self.optimizer, args.keep_every_n)
        # self.last_step = self.saver.restore(
        #     self.model_dir, map_to_cpu=args.restore_map_to_cpu,
        #     step=getattr(args, 'step', None))
        # if self.last_step == 0 and args.pretrained:
        #     for kind_path in args.pretrained.split(':_:'):
        #         kind, path = kind_path.split('::')
        #         self.load_pretrained(kind, path)

    def load_pretrained(self, kind, path):
        if kind == 'entire-model':
            keep_weight = lambda x: True
        elif kind == 'encoder':
            keep_weight = lambda x: \
                {'encoder': True, 'code_encoder': True, 'decoder': False,
                 'optimizer': False}[
                    x.split(".")[0]]
        else:
            raise NotImplementedError

        step = self.saver.restore(path, map_to_cpu=self.args.restore_map_to_cpu,
                                  step=self.args.pretrained_step,
                                  keep_weight=keep_weight)
        assert step == self.args.pretrained_step, "Step {} of model {} does not work".format(
            path,
            self.args.pretrained_step)

    def compute_loss(self, batch):
        raise NotImplementedError

    def inference(self, batch):
        raise NotImplementedError

    def debug(self, batch):
        raise NotImplementedError

    def train(self, batch):
        self.update_lr()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        if self.gradient_clip is not None and self.gradient_clip > 0:
            nn.utils.clip_grad_norm(self.model.parameters(),
                                    self.gradient_clip)
        self.optimizer.step()
        self.last_step += 1
        # if self.debug_every_n > 0 and self.last_step % self.debug_every_n == 0:
        #     self.debug(batch)
        # if self.last_step % self.save_every_n == 0:
        #     self.saver.save(self.model_dir, self.last_step)
        return {'loss': loss.data.item()}

    def eval(self, batch):
        results = self.inference(batch)
        correct = 0
        for example, res in zip(batch, results):
            if example.code_sequence == res.code_sequence or example.code_tree == res.code_tree:
                correct += 1
        return {'correct': correct, 'total': len(batch)}

    def update_lr(self):
        if self.lr_decay_steps is None or self.lr_decay_rate is None:
            return

        lr = self.lr * self.lr_decay_rate ** (self.last_step //
                                              self.lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def batch_processor(self, for_eval):
        '''Returns a function used to process batched data for this class.'''

        def default_processor(batch):
            return batch

        return default_processor


class BaseCodeModel(BaseModel):
    def __init__(self, cuda,
                 learning_rate,
                 lr_decay_steps,
                 lr_decay_rate,
                 gradient_clip,
                 ):
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.lr = learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.gradient_clip = gradient_clip

        super(BaseCodeModel, self).__init__()

        if cuda:
            self.model.cuda()
        print(self.model)

    # def _try_sequences(self, vocab, sequences, batch, beam_size):
    #     result = [[] for _ in range(len(batch))]
    #     counters = [0 for _ in range(len(batch))]
    #     candidates = [[] for _ in range(len(batch))]
    #     max_eval_trials = self.args.max_eval_trials or beam_size
    #     for batch_id, outputs in enumerate(sequences):
    #         example = batch[batch_id]
    #         # print("===", example.code_tree)
    #         candidates[batch_id] = [[vocab.itos(idx) for idx in ids]
    #                                 for ids in outputs]
    #         for code in candidates[batch_id][:max_eval_trials]:
    #             counters[batch_id] += 1
    #             stats = executor.evaluate_code(
    #                 code, example.schema.args, example.input_tests,
    #                 self.executor.execute)
    #             ok = (stats['correct'] == stats['total'])
    #             # print(code, stats)
    #             if ok:
    #                 result[batch_id] = code
    #                 break
    #     return [InferenceResult(code_sequence=seq,
    #                             info={'trees_checked': c, 'candidates': cand})
    #             for seq, c, cand in zip(result, counters, candidates)]


class BaseKarelModel(BaseCodeModel):
    pass

    # def eval(self, batch):
    #     results = self.inference(batch)
    #     correct = 0
    #     code_seqs = batch.code_seqs.cpu()
    #     for code_seq, res in zip(code_seqs, results):
    #         code_tokens = code_to_tokens(list(np.array(code_seq.data[1:])), self.vocab)
    #         if code_tokens == res.code_sequence:
    #             correct += 1
    #     return {'correct': correct, 'total': len(code_seqs)}

    def _try_sequences(self, vocab, sequences, input_grids, output_grids,
                       beam_size):
        result = [[] for _ in range(len(sequences))]
        counters = [0 for _ in range(len(sequences))]
        candidates = [[] for _ in range(len(sequences))]
        # max_eval_trials = self.max_eval_trials or beam_size
        for batch_id, outputs in enumerate(sequences):
            input_tests = [
                {
                    'input': np.where(inp.numpy().ravel())[0].tolist(),
                    'output': np.where(out.numpy().ravel())[0].tolist(),
                }
                for inp, out in zip(
                    torch.split(input_grids[batch_id].data.cpu(), 1),
                    torch.split(output_grids[batch_id].data.cpu(), 1), )
            ]
            candidates[batch_id] = [[vocab.itos(idx) for idx in ids]
                                    for ids in outputs]
            # for code in candidates[batch_id][:max_eval_trials]:
            #     counters[batch_id] += 1
            #     stats = executor.evaluate_code(code, None, input_tests,
            #                                    self.executor.execute)
            #     ok = (stats['correct'] == stats['total'])
            #     if ok:
            #         result[batch_id] = code
            #         break
        return None  # [
        #     InferenceResult(
        #         code_sequence=seq,
        #         info={'trees_checked': c,
        #               'candidates': cand})
        #     for seq, c, cand in zip(result, counters, candidates)
        # ]


class KarelLGRLRefineModel(BaseKarelModel):
    def __init__(self, vocab, karel_hidden_size, cuda, learning_rate,
                 lr_decay_steps, lr_decay_rate, gradient_clip,
                 max_beam_trees,
                 max_decoder_length,
                 use_length_penalty,
                 length_penalty_factor,
                 seq_max_len=45,
                 ):
        self.vocab = vocab
        self.model = LGRLRefineKarel(vocab_size, karel_hidden_size, cuda)
        self.cuda = cuda
        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.max_beam_trees = max_beam_trees
        self.max_decoder_length = max_decoder_length
        self.use_length_penalty = use_length_penalty
        self.length_penalty_factor = length_penalty_factor

        self.max_len = seq_max_len

        if cuda:
            self.model = self.model.cuda()

        self.trace_grid_lengths = []
        self.trace_event_lengths = []
        self.trace_lengths = []
        super(KarelLGRLRefineModel, self).__init__(cuda, learning_rate,
                                                   lr_decay_steps, lr_decay_rate,
                                                   gradient_clip)

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def load_model(self, best_models_path):
        self.model.load_state_dict(torch.load(f"{best_models_path}/debugger.pth"))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), f"{save_path}/debugger.pth")

    def compute_loss(self, input_tuple):
        input_grids, output_grids, code_seqs, dec_data, \
        ref_code, ref_trace_grids, ref_trace_events, \
        cag_interleave, orig_examples = input_tuple
        # TODO before the policy gradient this was impossible to execute since `orig_examples` is None whenever
        # this is not for_eval. Excluding the policy gradient
        if orig_examples and not self.args.train_policy_gradient_loss:
            for i, orig_example in enumerate(orig_examples):
                self.trace_grid_lengths.append((orig_example.idx, [
                    ref_trace_grids.lengths[ref_trace_grids.sort_to_orig[i * 5
                                                                         + j]]
                    for j in range(5)
                ]))
                self.trace_event_lengths.append((orig_example.idx, [
                    len(ref_trace_events.interleave_indices[i * 5 + j])
                    for j in range(5)
                ]))
                self.trace_lengths.append(
                    (orig_example.idx, np.array(self.trace_grid_lengths[-1][1])
                     + np.array(self.trace_event_lengths[-1][1])))

        if self.cuda:
            input_grids = input_grids.cuda(non_blocking=True)
            output_grids = output_grids.cuda(non_blocking=True)
            code_seqs = maybe_cuda(code_seqs, non_blocking=True)
            dec_data = maybe_cuda(dec_data, non_blocking=True)
            ref_code = maybe_cuda(ref_code, non_blocking=True)
            ref_trace_grids = maybe_cuda(ref_trace_grids, non_blocking=True)
            ref_trace_events = maybe_cuda(ref_trace_events, non_blocking=True)

        # io_embeds shape: batch size x num pairs (5) x hidden size (512)
        io_embed, ref_code_memory, ref_trace_memory = self.model.encode(
            input_grids, output_grids, ref_code, ref_trace_grids,
            ref_trace_events, cag_interleave)

        return self.calculate_supervised_loss(io_embed, ref_code_memory,
                                              ref_trace_memory, code_seqs,
                                              dec_data)

    def calculate_supervised_loss(self, io_embed, ref_code_memory,
                                  ref_trace_memory, code_seqs,
                                  dec_data):
        logits, labels = self.model.decode(io_embed, ref_code_memory,
                                           ref_trace_memory, code_seqs,
                                           dec_data)
        return self.criterion(
            logits.view(-1, logits.shape[-1]), labels.contiguous().view(-1))

    def debug(self, batch):
        code = code_to_tokens(batch.code_seqs.data[0, 1:], self.vocab)
        print("Code: %s" % ' '.join(code))

    def beam_sample(self, input_grids, output_grids, ref_code, ref_trace_grids,
                    ref_trace_events, cag_interleave, top_k):
        io_embed, ref_code_memory, ref_trace_memory = self.model.encode(
            input_grids, output_grids, ref_code, ref_trace_grids,
            ref_trace_events, cag_interleave)
        init_state = self.model.decoder.init_state(
            ref_code_memory, ref_trace_memory,
            io_embed.shape[0], io_embed.shape[1])
        memory = self.model.decoder.prepare_memory(io_embed, ref_code_memory,
                                                   ref_trace_memory, ref_code)

        sequences = beam_search.beam_search(
            len(input_grids),
            init_state,
            memory,
            self.model.decode_token,
            self.max_beam_trees,
            cuda=self.cuda,
            max_decoder_length=self.max_decoder_length,
            return_attention=False,
            return_beam_search_result=False,
            differentiable=False,
            use_length_penalty=self.use_length_penalty,
            factor=self.length_penalty_factor)

        # TODO: pick top k

        return sequences

    def get_evaluation_data(self, batch, tgt_seq, nb_rollouts=100,
                            top_k=(5,)):  # topk a list
        input_grids, output_grids, _1, dec_data, ref_code, \
        ref_trace_grids, ref_trace_events, cag_interleave, _2 = batch
        if self.cuda:
            input_grids = input_grids.cuda(non_blocking=True)
            output_grids = output_grids.cuda(non_blocking=True)
            dec_data = maybe_cuda(dec_data, non_blocking=True)
            ref_code = maybe_cuda(ref_code, non_blocking=True)
            ref_trace_grids = maybe_cuda(ref_trace_grids, non_blocking=True)
            ref_trace_events = maybe_cuda(ref_trace_events, non_blocking=True)

        io_embed, ref_code_memory, ref_trace_memory = self.model.encode(
            input_grids, output_grids, ref_code, ref_trace_grids,
            ref_trace_events, cag_interleave)
        init_state = self.model.decoder.init_state(
            ref_code_memory, ref_trace_memory,
            io_embed.shape[0], io_embed.shape[1])
        memory = self.model.decoder.prepare_memory(io_embed, ref_code_memory,
                                                   ref_trace_memory, ref_code)

        pre_paths = beam_search.beam_search(len(input_grids),
                                            init_state,
                                            memory,
                                            self.model.decode_token,
                                            self.max_beam_trees,
                                            cuda=self.cuda,
                                            max_decoder_length=self.max_decoder_length,
                                            return_attention=False,
                                            return_beam_search_result=False,
                                            differentiable=False,
                                            use_length_penalty=self.use_length_penalty,
                                            factor=self.length_penalty_factor)

        pre_paths = heapq.nlargest(top_k[0], pre_paths[0],
                                   key=lambda x: x[0])
        pre_paths = [[y[1] for y in pre_paths]]
        pre_paths = [self.model.decoder.postprocess_output(pre_paths, memory)]

        top_k_paths = []
        for pre_path in pre_paths:
            paths = []
            for path in pre_path:
                path = [x + [self.vocab['END']] for x in path]
                # path.append(self.vocab['END'])  # TODO: to test exact match
                path = [F.pad(torch.IntTensor(x), (0, self.max_len - len(x)),
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

    def inference(self, input_tuple):
        input_grids, output_grids, _1, dec_data, ref_code, \
        ref_trace_grids, ref_trace_events, cag_interleave, _2 = input_tuple
        if self.cuda:
            input_grids = input_grids.cuda(non_blocking=True)
            output_grids = output_grids.cuda(non_blocking=True)
            dec_data = maybe_cuda(dec_data, non_blocking=True)
            ref_code = maybe_cuda(ref_code, non_blocking=True)
            ref_trace_grids = maybe_cuda(ref_trace_grids, non_blocking=True)
            ref_trace_events = maybe_cuda(ref_trace_events, non_blocking=True)

        io_embed, ref_code_memory, ref_trace_memory = self.model.encode(
            input_grids, output_grids, ref_code, ref_trace_grids,
            ref_trace_events, cag_interleave)
        init_state = self.model.decoder.init_state(
            ref_code_memory, ref_trace_memory,
            io_embed.shape[0], io_embed.shape[1])
        memory = self.model.decoder.prepare_memory(io_embed, ref_code_memory,
                                                   ref_trace_memory, ref_code)

        sequences = beam_search.beam_search(
            len(input_grids),
            init_state,
            memory,
            self.model.decode_token,
            self.max_beam_trees,
            cuda=self.cuda,
            max_decoder_length=self.max_decoder_length,
            return_attention=False,
            return_beam_search_result=False,
            differentiable=False,
            use_length_penalty=self.use_length_penalty,
            factor=self.length_penalty_factor)

        sequences = self.model.decoder.postprocess_output(sequences, memory)

        return self._try_sequences(self.vocab, sequences, input_grids,
                                   output_grids, self.max_beam_trees)

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

    def _map_code(self, code):
        return [self.vocab[t] for t in code]

    def _translate_code(self, code):
        traversed_code_tokens = np.array(
            self._map_code(traverse_pre_order(code, teacher_code=True)))
        traversed_code_tokens = np.append(traversed_code_tokens, 0)

        return traversed_code_tokens
