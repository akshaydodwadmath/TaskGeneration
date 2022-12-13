import collections

import numpy as np
import torch
from src.agent.debugger_agent import beam_search
from torch import nn
from torch.autograd import Variable

IMG_DIM = 18
IMG_SIZE = torch.Size((16, IMG_DIM, IMG_DIM))

SequenceMemory = collections.namedtuple('SequenceMemory', ['mem', 'state'])


class MaskedMemory(collections.namedtuple('MaskedMemory', ['memory',
                                                           'attn_mask'])):

    def expand_by_beam(self, beam_size):
        return MaskedMemory(*(v.unsqueeze(1).repeat(1, beam_size, *([1] * (
                v.dim() - 1))).view(-1, *v.shape[1:]) for v in self))

    def apply(self, fn):
        return MaskedMemory(fn(self.memory), fn(self.attn_mask))


class DoNotAugmentWithTrace(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.hs = hs

    def forward(self, inp_embed, *args, **kwargs):
        return inp_embed

    @property
    def output_embedding_size(self):
        return self.hs


def get_attn_mask(seq_lengths, cuda):
    max_length, batch_size = max(seq_lengths), len(seq_lengths)
    ranges = torch.arange(
        0, max_length,
        out=torch.LongTensor()).unsqueeze(0).expand(batch_size, -1)
    attn_mask = (ranges >= torch.LongTensor(seq_lengths).unsqueeze(1))
    if cuda:
        attn_mask = attn_mask.cuda()
    return attn_mask


def lstm_init(cuda, num_layers, hidden_size, *batch_sizes):
    init_size = (num_layers,) + batch_sizes + (hidden_size,)
    init = Variable(torch.zeros(*init_size))
    if cuda:
        init = init.cuda()
    return (init, init)


def default(value, if_none):
    return if_none if value is None else value


def expand(v, k):
    # Input: d1 x ...
    # Output: d1 * k x ... where
    #   out[0] = out[1] = ... out[k],
    #   out[k + 0] = out[k + 1] = ... out[k + k],
    # and so on.
    return v.unsqueeze(1).repeat(1, k, *([1] *
                                         (v.dim() - 1))).view(-1, *v.shape[1:])


def unexpand(v, k):
    # Input: d1 x ...
    # Output: d1 / k x k x ...
    return v.view(-1, k, *v.shape[1:])


def flatten(v, k):
    # Input: d1 x ... x dk x dk+1 x ... x dn
    # Output: d1 x ... x dk * dk+1 x ... x dn
    args = v.shape[:k] + (-1,) + v.shape[k + 2:]
    return v.contiguous().view(*args)


def maybe_concat(items, dim=None):
    to_concat = [item for item in items if item is not None]
    if not to_concat:
        return None
    elif len(to_concat) == 1:
        return to_concat[0]
    else:
        return torch.cat(to_concat, dim)


def take(tensor, indices):
    '''Equivalent of numpy.take for Torch tensors.'''
    indices_flat = indices.contiguous().view(-1)
    return tensor[indices_flat].view(indices.shape + tensor.shape[1:])


def set_pop(s, value):
    if value in s:
        s.remove(value)
        return True
    return False


class CodeEncoder(nn.Module):
    def __init__(self, vocab_size,
                 hs,
                 cuda=True):
        super(CodeEncoder, self).__init__()

        self._cuda = cuda
        self.karel_hidden_size = hs
        # corresponds to self.token_embed
        self.embed = nn.Embedding(vocab_size, hs)
        self.augment_with_trace = DoNotAugmentWithTrace(hs)

        self.encoder = nn.LSTM(
            input_size=self.augment_with_trace.output_embedding_size,
            hidden_size=hs,
            num_layers=2,
            bidirectional=True,
            batch_first=True)

    def forward(self, inputs, input_grid, output_grid, traces, trace_events):
        # inputs: PackedSequencePlus, batch size x sequence length
        inp_embed = inputs.apply(self.embed)
        inp_embed = self.augment_with_trace(inp_embed, input_grid, output_grid, traces,
                                            trace_events, list(inputs.orig_lengths()))
        # output: PackedSequence, batch size x seq length x hidden (256 * 2)
        # state: 2 (layers) * 2 (directions) x batch x hidden size (256)
        output, state = self.encoder(inp_embed.ps,
                                     lstm_init(self._cuda, 4,
                                               self.karel_hidden_size,
                                               inp_embed.ps.batch_sizes[0]))

        return SequenceMemory(
            inp_embed.with_new_ps(output),
            state)


class LGRLSeqRefineEditDecoder(nn.Module):
    def __init__(self, vocab_size, cuda, hs):
        super(LGRLSeqRefineEditDecoder, self).__init__()
        self._cuda = cuda

        num_ops = 4 + 2 * vocab_size

        self.karel_hidden_size = hs
        self.use_code_memory = True

        self.op_embed = nn.Embedding(num_ops, hs)
        self.last_token_embed = nn.Embedding(vocab_size, hs)
        self.decoder = nn.LSTM(
            input_size=hs * 2 + hs * 2 + hs * 2,
            hidden_size=hs,
            num_layers=2)
        self.out = nn.Linear(hs, num_ops, bias=False)

        # Given the past op, whether to increment source_loc or not.
        # Don't increment for <s>, </s>, insert ops
        self.increment_source_loc = np.ones(num_ops, dtype=int)
        self.increment_source_loc[:2] = 0
        self.increment_source_loc[range(4, num_ops, 2)] = 0

        # If we have exhausted all source tokens, then we allow </s> and
        # insertion only.
        # In all cases, don't allow <s>.
        # Positions where mask is 1 will get -inf as the logit.
        self.end_mask = torch.BoolTensor([
            [1] + [0] * (num_ops - 1),
            # <s>, </s>, keep, delete
            np.concatenate(([1, 0, 1, 1], self.increment_source_loc[4:])).tolist()
        ])
        if self._cuda:
            self.end_mask = self.end_mask.cuda()

        # if self.use_code_state:
        #     self.state_h_proj = torch.nn.ModuleList(
        #         [nn.Linear(hs * 2, hs) for _ in range(2)])
        #     self.state_c_proj = torch.nn.ModuleList(
        #         [nn.Linear(hs * 2, hs) for _ in range(2)])

    def prepare_memory(self, io_embed, code_memory, _, ref_code):
        # code_memory:
        #   SequenceMemory, containing:
        #     mem: PackedSequencePlus, batch size x code length x 512
        #     state: tuple containing two of
        #       2 (layers) * 2 (directions) x batch size x 256
        #   or None
        pairs_per_example = io_embed.shape[1]
        ref_code = ref_code.cpu()
        # if self.use_code_attn:
        #     # batch x code length x 512
        #     code_memory, code_lengths = code_memory.mem.pad(batch_first=True)
        #     code_mask = get_attn_mask(code_lengths, self._cuda)
        #     code_memory = code_memory.unsqueeze(1).repeat(1, pairs_per_example,
        #                                                   1, 1)
        #     code_mask = code_mask.unsqueeze(1).repeat(1, pairs_per_example, 1)
        #     code_memory = MaskedMemory(code_memory, code_mask)
        # else:
        code_memory = code_memory.mem

        return LGRLRefineEditMemory(io_embed, code_memory, ref_code)

    def forward(self, io_embed, code_memory, _1, _2, dec_data):

        dec_output, dec_data = self.common_forward(io_embed, code_memory, _1, _2,
                                                   dec_data)

        # if self.use_code_attn:
        #     logits = torch.cat(dec_output, dim=0)
        #     labels = dec_data.output.ps.data
        #     return logits, labels
        #
        # else:
        logits = self.out(dec_output)
        labels = dec_data.output.ps.data

        return logits, labels

    def common_forward(self, io_embed, code_memory, _1, _2, dec_data):
        # io_embed: batch size x num pairs x 512
        # code_memory:
        #   PackedSequencePlus, batch size x code length x 512
        #   or None
        # dec_input:
        #   PackedSequencePlus, batch size x num ops x 4
        #      (op, emb pos, last token, io_embed index)
        # dec_output:
        #   PackedSequencePlus, batch size x num ops x 1 (op)
        batch_size, pairs_per_example = io_embed.shape[:2]

        # if self.use_code_attn:
        #     state = self.init_state(code_memory, None, batch_size,
        #                             pairs_per_example)
        #     memory = self.prepare_memory(
        #         io_embed, code_memory, None, dec_data.ref_code)
        #     # dec_input = dec_data.input.pad(batch_first=False)
        #     # for t in range(dec_input.shape[0]):
        #     #    dec_input_t = dec_input[t]
        #     #    self.decode_token(dec_input_t[2])
        #
        #     memory.io = memory.io[list(dec_data.input.orig_to_sort)]
        #     memory.code = memory.code.apply(
        #         lambda t: t[list(dec_data.input.orig_to_sort)])
        #     # lambda t: t[[
        #     #    exp_i
        #     #    for i in dec_data.input.orig_to_sort
        #     #    for exp_i in range(
        #     #        i * pairs_per_example,
        #     #        i * pairs_per_example + pairs_per_example)
        #     # ]])
        #
        #     logits = []
        #     offset = 0
        #     last_bs = 0
        #     batch_order = dec_data.input.orig_to_sort
        #     for i, bs in enumerate(dec_data.input.ps.batch_sizes):
        #         # Shape: bs x
        #         dec_data_slice = dec_data.input.ps.data[offset:offset + bs]
        #         if bs < last_bs:
        #             memory.io = memory.io[:bs]
        #             memory.code = memory.code.apply(lambda t: t[:bs])
        #             batch_order = batch_order[:bs]
        #             state = state.truncate(bs)
        #
        #         state, logits_for_t = self.decode_token(
        #             dec_data_slice[:, 0], state, memory, None, batch_order,
        #             use_end_mask=False)
        #         logits.append(logits_for_t)
        #         offset += bs
        #         last_bs = bs
        #
        #     return logits, dec_data

        io_embed_flat = io_embed.view(-1, *io_embed.shape[2:])

        dec_input = dec_data.input.apply(lambda d: maybe_concat(
            [
                self.op_embed(d[:, 0]),  # 256
                code_memory.mem.ps.data[d[:, 1]] if self.use_code_memory else None,
                # 512
                self.last_token_embed(d[:, 2])],  # 256
            dim=1)).expand(pairs_per_example)

        dec_input = dec_input.apply(
            lambda d: torch.cat([d,
                                 io_embed_flat[dec_data.io_embed_indices]], dim=1))

        state = self.init_state(code_memory, None, batch_size,
                                pairs_per_example)
        state = (flatten(state.h, 1), flatten(state.c, 1))

        dec_output, _ = self.decoder(dec_input.ps, state)
        dec_output, _ = dec_output.data.view(-1, pairs_per_example,
                                             *dec_output.data.shape[1:]).max(dim=1)

        return dec_output, dec_data

    def rl_forward(self, io_embed, code_memory, _1, _2, dec_data):
        dec_output, dec_data = self.common_forward(io_embed, code_memory, _1, _2,
                                                   dec_data)

        if self.use_code_attn:
            logits = torch.cat(dec_output, dim=0)
            labels = dec_data.output.ps.data
            return logits, labels, None

        else:
            logits = self.out(dec_output)
            labels = dec_data.output.ps.data

        return logits, labels, dec_output, (dec_data.output.ps.batch_sizes,
                                            dec_data.output.orig_to_sort)  # dec_data.output.lengths

    def decode_token(self, token, state, memory, attentions, batch_order=None,
                     use_end_mask=True, return_dec_out=False):
        pairs_per_example = memory.io.shape[1]
        token_np = token.data.cpu().numpy()
        new_finished = state.finished.copy()

        # token shape: batch size (* beam size)
        # op_emb shape: batch size (* beam size) x 256
        op_emb = expand(self.op_embed(token), pairs_per_example)

        # Relevant code_memory:
        # - same as last code_memory if we have insert, <s>, </s>
        # - incremented otherwise
        new_source_locs = state.source_locs + self.increment_source_loc[
            token_np]
        new_source_locs[state.finished] = 0

        if self.use_code_memory:
            code_memory_indices = memory.code.raw_index(
                orig_batch_idx=range(len(state.source_locs)),
                seq_idx=new_source_locs)
            code_memory = memory.code.ps.data[code_memory_indices.tolist()]
            code_memory = expand(code_memory, pairs_per_example)
        else:
            code_memory = None

        # Last token embedings
        # keep: token from appropriate source location
        #   (state.source_locs, not new_source_locs)
        # delete: UNK (ID 2)
        # insert, replace: the inserted/replaced token
        last_token_indices = np.zeros(len(token_np), dtype=int)
        keep_indices = []
        keep_orig_batch_indices = []
        keep_source_locs = []

        if batch_order is None:
            batch_order = range(len(token_np))
        for i, (batch_idx, t) in enumerate(zip(batch_order, token_np)):
            if t == 0:  # <s>
                last_token_indices[i] = t
            elif t == 1:  # </s>
                last_token_indices[i] = t
                new_finished[i] = True
            elif t == 2:  # keep
                keep_indices.append(i)
                keep_orig_batch_indices.append(batch_idx)
                keep_source_locs.append(state.source_locs[i])

                # last_token_indices.append(
                #        memory.ref_code.select(batch_idx,
                #            state.source_locs[batch_idx]).data[0])
                # last_token_indices.append(memory.source_tokens[batch_idx][
                #    state.source_locs[batch_idx]])
            elif t == 3:  # delete
                last_token_indices[i] = 2  # UNK
            elif t >= 4:
                last_token_indices[i] = (t - 4) // 2
            else:
                raise ValueError(t)
        if keep_indices:
            last_token_indices[keep_indices] = memory.ref_code.select(
                keep_orig_batch_indices,
                [state.source_locs[i] for i in keep_indices]).data.numpy()

        last_token_indices = Variable(
            torch.LongTensor(last_token_indices))
        if self._cuda:
            last_token_indices = last_token_indices.cuda()
        last_token_emb = self.last_token_embed(last_token_indices)
        last_token_emb = expand(last_token_emb, pairs_per_example)

        dec_input = maybe_concat([op_emb, code_memory, last_token_emb,
                                  flatten(memory.io, 0)], dim=1)

        dec_output, new_state = self.decoder(
            # 1 (time) x batch (* beam) * num pairs x hidden size
            dec_input.unsqueeze(0),
            # v before: 2 x batch (* beam) x num pairs x hidden
            # v after:  2 x batch (* beam) * num pairs x hidden
            (flatten(state.h, 1), flatten(state.c, 1)))
        new_state = (new_state[0].view_as(state.h),
                     new_state[1].view_as(state.c))

        # shape after squeezing: batch size (* beam size) * num pairs x hidden
        dec_output = dec_output.squeeze(0)
        # if self.use_code_attn:
        #     new_context, _ = self.code_attention(
        #         dec_output,
        #         flatten(memory.code.memory, 0),
        #         flatten(memory.code.attn_mask, 0))
        #     dec_output = maybe_concat([dec_output, new_context], dim=1)
        #     new_context = new_context.view(
        #         -1, pairs_per_example, new_context.shape[-1])
        # else:
        new_context = None

        dec_output = dec_output.view(-1, pairs_per_example,
                                     *dec_output.shape[1:])
        dec_output, _ = dec_output.max(dim=1)

        # batch (* beam) x vocab size
        logits = self.out(dec_output)

        # If we have depleted the source tokens, then all we can do is end the
        # output or insert new tokens.
        # XXX use_end_mask must be false if state/tokens isn't in the same batch
        # order as memory.ref_code.
        if use_end_mask:
            end_mask = []
            for i, (loc, source_len) in enumerate(zip(new_source_locs,
                                                      memory.ref_code.orig_lengths())):
                if state.finished[i]:
                    end_mask.append(1)
                # source_len is 1 longer than in reality because we appended
                # </s> to each source sequence.
                elif loc == source_len - 1 or source_len == 1:
                    end_mask.append(1)
                elif loc >= source_len:
                    print(
                        "Warning: loc ({}) >= source_len ({})".format(loc, source_len))
                    end_mask.append(1)
                else:
                    end_mask.append(0)
            logits.data.masked_fill_(self.end_mask[end_mask], float('-inf'))

        if return_dec_out:
            return LGRLRefineEditDecoderState(new_source_locs, new_finished,
                                              new_context,
                                              *new_state), logits, dec_output
        else:
            return LGRLRefineEditDecoderState(new_source_locs, new_finished,
                                              new_context, *new_state), logits

    def postprocess_output(self, sequences, memory):
        ref_code = memory.ref_code.cpu()

        result = []

        ref_code_insert_locs = []
        ref_code_batch_indices = []
        ref_code_seq_indices = []

        for batch_idx, beam_outputs in enumerate(sequences):
            processed_beam_outputs = []
            for ops in beam_outputs:
                real_tokens = []
                source_index = 0
                for op in ops:
                    if op == 2:  # keep
                        ref_code_insert_locs.append(
                            (len(result), len(processed_beam_outputs),
                             len(real_tokens)))
                        ref_code_batch_indices.append(batch_idx)
                        ref_code_seq_indices.append(source_index)

                        real_tokens.append(-1)
                        source_index += 1
                    elif op == 3:  # delete
                        source_index += 1
                    elif op >= 4:
                        t = (op - 4) // 2
                        is_replace = (op - 4) % 2
                        real_tokens.append(t)
                        source_index += is_replace
                    else:
                        raise ValueError(op)
                processed_beam_outputs.append(real_tokens)
            result.append(processed_beam_outputs)

        if ref_code_insert_locs:
            tokens = ref_code.select(
                ref_code_batch_indices,
                ref_code_seq_indices).data.numpy()
            for (b, p, s), t in zip(ref_code_insert_locs, tokens):
                result[b][p][s] = t

        return result

    def init_state(self, ref_code_memory, ref_trace_memory,
                   batch_size, pairs_per_example):
        source_locs = np.zeros(batch_size, dtype=int)
        finished = np.zeros(batch_size, dtype=bool)

        # if self.use_code_attn:
        #     context_size = (batch_size, pairs_per_example, 512)
        #     context = Variable(torch.zeros(*context_size))
        #     if self._cuda:
        #         context = context.cuda()
        # else:
        context = None

        # if self.use_code_state:
        #     new_state = []
        #     for s, proj in zip(ref_code_memory.state,
        #                        (self.state_h_proj, self.state_c_proj)):
        #         # Shape: layers (2) x directions (2) x batch x hidden size
        #         s = s.contiguous().view(-1, 2, *s.shape[1:])
        #         # Shape: layers (2) x batch x directions (2) x hidden size
        #         s = s.permute(0, 2, 1, 3)
        #         # Shape: layers (2) x batch x directions * hidden size
        #         s = s.contiguous().view(*(s.shape[:2] + (-1,)))
        #         new_s = []
        #         for s_layer, proj_layer in zip(s, proj):
        #             # Input: batch x directions * hidden size (=512)
        #             # Output: batch x 256
        #             new_s.append(proj_layer(s_layer))
        #         # Shape: 2 x batch x 256
        #         new_s = torch.stack(new_s)
        #         # Shape: 2 x batch x num pairs x 256
        #         new_state.append(
        #             new_s.unsqueeze(2).repeat(1, 1, pairs_per_example, 1))
        #     return LGRLRefineEditDecoderState(
        #         source_locs,
        #         finished,
        #         context,
        #         *new_state)

        return LGRLRefineEditDecoderState(
            source_locs,
            finished,
            context,
            *lstm_init(self._cuda, 2, self.karel_hidden_size, batch_size,
                       pairs_per_example))


def maybe_mask(attn, attn_mask):
    if attn_mask is not None:
        assert attn_mask.size() == attn.size(), \
            'Attention mask shape {} mismatch ' \
            'with Attention logit tensor shape ' \
            '{}.'.format(attn_mask.size(), attn.size())

        attn.data.masked_fill_(attn_mask, -float('inf'))


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dim, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(dim, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        maybe_mask(attn, attn_mask)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class SimpleSDPAttention(ScaledDotProductAttention):
    def __init__(self, query_dim, values_dim, dropout_p=0.0):
        super(SimpleSDPAttention, self).__init__(values_dim, dropout_p)
        self.query_proj = nn.Linear(query_dim, values_dim)

    def forward(self, query, values, attn_mask=None):
        # query shape: batch x query dim
        # values shape: batch x num values x values dim
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        output, attn = super(SimpleSDPAttention, self).forward(
            self.query_proj(query).unsqueeze(1), values, values, attn_mask)

        output = output.squeeze(1)
        return output, attn


class LGRLRefineEditMemory(beam_search.BeamSearchMemory):
    __slots__ = ('io', 'code', 'ref_code')

    def __init__(self, io, code, ref_code):
        # io: batch (* beam size) x num pairs x hidden size
        self.io = io
        # code: PackedSequencePlus, batch (* beam size) x code length x 512
        self.code = code
        # ref_code: PackedSequencePlus, batch x seq length
        self.ref_code = ref_code

    def expand_by_beam(self, beam_size):
        io_exp = expand(self.io, beam_size)
        if self.code is None:
            code_exp = None
        elif isinstance(self.code, MaskedMemory):
            code_exp = self.code.expand_by_beam(beam_size)
        else:
            code_exp = self.code.expand(beam_size)
        ref_code_exp = self.ref_code.expand(beam_size)
        # source_tokens_exp = [
        #    tokens for tokens in self.source_tokens for _ in range(beam_size)
        # ]
        # return LGRLRefineMemory(io_exp, code_exp, source_tokens_exp)
        return LGRLRefineEditMemory(io_exp, code_exp, ref_code_exp)


class LGRLRefineEditDecoderState(
    collections.namedtuple('LGRLRefineEditDecoderState', [
        'source_locs', 'finished', 'context', 'h', 'c',
    ]),
    # source_locs: batch size (* beam size)
    beam_search.BeamSearchState):

    def select_for_beams(self, batch_size, indices):
        '''Return the hidden state necessary to continue the beams.

        batch size: int
        indices: 2 x batch size * beam size LongTensor
        '''
        indices_tuple = tuple(indices.data.numpy())
        selected = [
            self.source_locs.reshape(batch_size, -1)[indices_tuple],
            self.finished.reshape(batch_size, -1)[indices_tuple],
            None if self.context is None else self.context.view(
                batch_size, -1, *self.context.shape[1:])[indices_tuple]
        ]
        for v in self.h, self.c:
            # before: 2 x batch size (* beam size) x num pairs x hidden
            # after:  2 x batch size x beam size x num pairs x hidden
            v = v.view(2, batch_size, -1, *v.shape[2:])
            # result: 2 x indices.shape[1] x num pairs x hidden
            selected.append(v[(slice(None),) + indices_tuple])
        return LGRLRefineEditDecoderState(*selected)

    def truncate(self, k):
        return LGRLRefineEditDecoderState(
            self.source_locs[:k],
            self.finished[:k],
            self.context[:k],
            self.h[:, :k],
            self.c[:, :k])


class LGRLTaskEncoder(nn.Module):
    '''Implements the encoder from:

    Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis
    https://openreview.net/forum?id=H1Xw62kRZ
    '''

    def __init__(self, karel_hidden_size):
        super(LGRLTaskEncoder, self).__init__()

        self.input_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=IMG_SIZE[0], out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(), )
        self.output_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=IMG_SIZE[0], out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(), )

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(), )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(), )

        self.fc = nn.Linear(64 * 18 * 18, karel_hidden_size * 2)

    def forward(self, input_grid, output_grid):
        batch_dims = input_grid.shape[:-3]
        # flattened_batch_dims = math.prod(batch_dims)
        input_grid = input_grid.contiguous().view(-1, *input_grid.shape[-3:])
        output_grid = output_grid.contiguous().view(-1, *output_grid.shape[-3:])

        input_enc = self.input_encoder(input_grid)
        output_enc = self.output_encoder(output_grid)
        enc = torch.cat([input_enc, output_enc], 1)
        enc = enc + self.block_1(enc)
        enc = enc + self.block_2(enc)

        enc = self.fc(enc.view(*(batch_dims + (-1,))))
        return enc


class LGRLRefineKarel(nn.Module):
    def __init__(self, vocab_size, karel_hidden_size, cuda):
        super(LGRLRefineKarel, self).__init__()

        self.code_encoder = CodeEncoder(vocab_size, karel_hidden_size, cuda=cuda)

        self.decoder = LGRLSeqRefineEditDecoder(vocab_size, cuda, karel_hidden_size)

        # task_encoder
        self.encoder = LGRLTaskEncoder(karel_hidden_size)

    def encode(self, input_grid, output_grid, ref_code, ref_trace_grids,
               ref_trace_events, cag_interleave):
        # batch size x num pairs x 512
        io_embed = self.encoder(input_grid, output_grid)
        # PackedSequencePlus, batch size x length x 512
        ref_code_memory = self.code_encoder(ref_code, input_grid, output_grid,
                                            ref_trace_grids, ref_trace_events)
        # PackedSequencePlus, batch size x num pairs x length x  512
        # ref_trace_memory = self.trace_encoder(ref_code_memory, ref_trace_grids,
        #                                       ref_trace_events, cag_interleave)
        return io_embed, ref_code_memory, None  # ref_trace_memory

    def decode(self, io_embed, ref_code_memory, ref_trace_memory, outputs,
               dec_data):
        return self.decoder(io_embed, ref_code_memory, ref_trace_memory,
                            outputs, dec_data)

    def decode_token(self, token, state, memory, attentions=None):
        return self.decoder.decode_token(token, state, memory, attentions)
