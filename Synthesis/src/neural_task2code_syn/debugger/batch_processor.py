import collections
import itertools

import Levenshtein
import numpy as np
import torch
from src.agent.debugger_agent import prepare_spec
from src.agent.debugger_agent.debugger_agent import PackedDecoderData
from src.utils.code_linearizer import traverse_pre_order
from torch import nn
from torch.autograd import Variable


def compute_edit_ops(source_seq, target_seq, stoi):
    source_str = ''.join(chr(t) for t in source_seq)
    target_str = ''.join(chr(t) for t in target_seq)

    ops = Levenshtein.editops(source_str, target_str)
    i, op_idx = 0, 0
    while i < len(source_seq) or op_idx < len(ops):
        if op_idx == len(ops) or i < ops[op_idx][1]:
            yield i, 'keep', None
            i += 1
            continue
        op_type, source_pos, target_pos = ops[op_idx]
        op_idx += 1
        if op_type == 'insert':
            yield i, 'insert', target_seq[target_pos]
            continue
        elif op_type == 'replace':
            yield i, 'replace', target_seq[target_pos]
        elif op_type == 'delete':
            yield i, 'delete', None
        else:
            raise ValueError(op_type)
        i += 1


def compute_edit_ops_no_stoi(source_seq, target_seq):
    source_str = ''.join(chr(t) for t in source_seq)
    target_str = ''.join(chr(t) for t in target_seq)

    ops = Levenshtein.editops(source_str, target_str)
    i, op_idx = 0, 0
    while i < len(source_seq) or op_idx < len(ops):
        if op_idx == len(ops) or i < ops[op_idx][1]:
            yield (i, 'keep', None)
            i += 1
            continue
        op_type, source_pos, target_pos = ops[op_idx]
        op_idx += 1
        if op_type == 'insert':
            yield (i, 'insert', target_seq[target_pos])
            continue
        elif op_type == 'replace':
            yield (i, 'replace', target_seq[target_pos])
        elif op_type == 'delete':
            yield (i, 'delete', None)
        else:
            raise ValueError(op_type)
        i += 1


def denumpify(item):
    if type(item) in {tuple, list, np.ndarray}:  # no isinstance to avoid named tuples
        return [int(x) if np.issubdtype(type(x), np.integer) else x for x in item]
    else:
        return item


def lists_to_packed_sequence(lists, item_shape, tensor_type, item_to_tensor):
    # TODO: deduplicate with the version in prepare_spec.
    result = tensor_type(sum(len(lst) for lst in lists), *item_shape)

    sorted_lists, sort_to_orig, orig_to_sort = prepare_spec.sort_lists_by_length(lists)
    lengths = prepare_spec.lengths(sorted_lists)
    batch_bounds = prepare_spec.batch_bounds_for_packing(lengths)
    idx = 0
    for i, bound in enumerate(batch_bounds):
        for batch_idx, lst in enumerate(sorted_lists[:bound]):
            try:
                item_to_tensor(denumpify(lst[i]), batch_idx, result[idx])
            except:
                print(lst[i])
                print(type(lst[i]) == tuple)
                1 / 0
            idx += 1

    result = Variable(result)
    batch_bounds = torch.tensor(batch_bounds, dtype=torch.long)

    return prepare_spec.PackedSequencePlus(
        nn.utils.rnn.PackedSequence(result, batch_bounds),
        lengths, sort_to_orig, orig_to_sort)


KarelLGRLRefineExample = collections.namedtuple('KarelLGRLRefineExample', (
    'input_grids', 'output_grids', 'code_seqs', 'dec_data',
    'ref_code', 'ref_trace_grids', 'ref_trace_events',
    'cond_action_grid_interleave', 'orig_examples'))


def encode_grids_and_outputs(batch, seq, vocab):
    # TODO: concat the inps and outps
    # convert code to tokens
    inp_grids = [entry['in_grids'] for entry in batch]
    out_grids = [entry['out_grids'] for entry in batch]

    inp_grids = torch.stack(inp_grids)
    out_grids = torch.stack(out_grids)

    def translate_code(code):
        traversed_code_tokens = np.array(
            [vocab[t] for t in traverse_pre_order(code, teacher_code=True)])
        traversed_code_tokens = np.append(traversed_code_tokens, 0)
        return traversed_code_tokens

    lines = [
        [vocab.get('START', vocab.get('<S>', 0))] +
        translate_code(code).tolist() +
        [vocab.get('END', vocab.get('</S>', 0))]
        for code in seq]

    lens = [len(line) for line in lines]
    max_len = max(lens)

    tgt_lines = [line[0:] + [vocab.get('PAD', -1)] * (max_len - len(line)) for line in
                 lines]

    tgt_seq = torch.tensor(tgt_lines).long()

    return inp_grids, out_grids, tgt_seq


class KarelLGRLRefineBatchProcessor:
    def __init__(self, args, vocab, for_eval):
        self.args = args
        self.vocab = vocab
        self.for_eval = for_eval
        self.return_edits = False

    def __call__(self, batch, seq, mutated_seq):
        input_grids, output_grids, code_seqs = encode_grids_and_outputs(batch, seq,
                                                                        self.vocab)

        ref_code = prepare_spec.lists_to_packed_sequence(
            mutated_seq,
            self.vocab,
            False,
            volatile=False)

        dec_data = self.compute_edit_ops(seq, mutated_seq, ref_code, self.return_edits)

        ref_trace_grids, ref_trace_events = None, None
        cag_interleave = None

        orig_examples = None

        return KarelLGRLRefineExample(
            input_grids, output_grids, code_seqs, dec_data,
            ref_code, ref_trace_grids, ref_trace_events, cag_interleave,
            orig_examples)

    def compute_edit_ops(self, code, mutated_code, ref_code, return_edits=False):
        # Sequence length: 2 + len(edit_ops)
        #
        # Op encoding:
        #   0: <s>
        #   1: </s>
        #   2: keep
        #   3: delete
        #   4: insert vocab 0
        #   5: replace vocab 0
        #   6: insert vocab 1
        #   7: replace vocab 1
        #   ...
        #
        # Inputs to RNN:
        # - <s> + op
        # - emb from source position + </s>
        # - <s> + last generated token (or null if last action was deletion)
        #
        # Outputs of RNN:
        # - op + </s>

        def translate_code(code):
            traversed_code_tokens = np.array(
                [self.vocab[t] for t in traverse_pre_order(code, teacher_code=True)])
            traversed_code_tokens = np.append(traversed_code_tokens, 0)
            return traversed_code_tokens

        edit_lists = []
        for batch_idx, item in enumerate(code):
            mut = translate_code(mutated_code[batch_idx])
            tgt = translate_code(item)
            edit_ops = list(
                compute_edit_ops(mut, tgt, self.vocab))
            dest_iter = itertools.chain([0], tgt)  # START token

            # Op = <s>, emb location, last token = <s>
            source_locs, ops, values = [list(x) for x in zip(*edit_ops)]
            source_locs.append(len(mut))
            ops = [0] + ops
            values = [None] + values

            edit_list = []
            for source_loc, op, value in zip(source_locs, ops, values):
                if op == 'keep':
                    op_idx = 2
                elif op == 'delete':
                    op_idx = 3
                elif op == 'insert':
                    op_idx = 4 + 2 * value
                elif op == 'replace':
                    op_idx = 5 + 2 * value
                elif isinstance(op, int):
                    op_idx = op
                else:
                    raise ValueError(op)

                # Set last token to UNK if operation is delete
                # XXX last_token should be 0 (<s>) at the beginning
                try:
                    last_token = 2 if op_idx == 3 else next(dest_iter)
                except StopIteration:
                    raise Exception('dest_iter ended early')

                l = ref_code.lengths[ref_code.sort_to_orig[batch_idx]]
                assert source_loc < l
                edit_list.append((
                    op_idx, ref_code.raw_index(batch_idx, source_loc),
                    last_token))
            stopped = False
            try:
                next(dest_iter)
            except StopIteration:
                stopped = True
            assert stopped

            # Op = END, emb location and last token are irrelevant
            edit_list.append((1, None, None))
            edit_lists.append(edit_list)

        rnn_inputs = lists_to_packed_sequence(
            [lst[:-1] for lst in edit_lists], (3,), torch.LongTensor,
            lambda op_emb_pos_last_token, _, out:
            out.copy_(torch.LongTensor([*op_emb_pos_last_token])))
        rnn_outputs = lists_to_packed_sequence(
            [lst[1:] for lst in edit_lists], (1,), torch.LongTensor,
            lambda op_emb_pos_last_token, _, out:
            out.copy_(torch.LongTensor([op_emb_pos_last_token[0]])))

        io_embed_indices = torch.LongTensor([
            expanded_idx
            for b in rnn_inputs.ps.batch_sizes
            for orig_idx in rnn_inputs.orig_to_sort[:b]
            for expanded_idx in range(orig_idx * 5, orig_idx * 5 + 5)
        ])

        if return_edits:
            return (PackedDecoderData(rnn_inputs, rnn_outputs, io_embed_indices,
                                      ref_code), edit_lists)
        else:
            return PackedDecoderData(rnn_inputs, rnn_outputs, io_embed_indices,
                                     ref_code)

    def compute_edit_ops_no_char(self, batch, code_seqs, ref_code):
        # print([self.vocab._rev_vocab[int(token)] for cd in code_seqs for token in cd if token > 0])
        edit_lists = []
        for batch_idx, item in enumerate(zip(batch, code_seqs)):
            # Removed the previously made sos token and end token
            code_sequence = list(
                np.array(item[1])[np.array(item[1]) > -1])  # -1])[1:-1]
            # Double
            ref_example_code_sequence = list(np.array(item[0])[np.array(item[0]) > -1])
            edit_ops = list(
                compute_edit_ops_no_stoi(ref_example_code_sequence,
                                         code_sequence))
            dest_iter = itertools.chain(code_sequence)

            # Op = <s>, emb location, last token = <s>
            source_locs, ops, values = [list(x) for x in zip(*edit_ops)]
            # source_locs.append(len(ref_example_code_sequence))
            # ops = [0] + ops
            # values = [None] + values

            edit_list = []
            op_idx = 0
            for source_loc, op, value in zip(source_locs, ops, values):
                if op == 'keep':
                    op_idx = 2
                elif op == 'delete':
                    op_idx = 3
                elif op == 'insert':
                    op_idx = 4 + 2 * self.vocab.stoi(value)
                elif op == 'replace':
                    op_idx = 5 + 2 * self.vocab.stoi(value)
                elif isinstance(op, int):
                    op_idx = op
                else:
                    raise ValueError(op)

                # Set last token to UNK if operation is delete
                # XXX last_token should be 0 (<s>) at the beginning
                try:
                    if op_idx == 3:
                        last_token = 2
                    else:
                        last_token = next(dest_iter)
                except StopIteration:
                    raise Exception('dest_iter ended early')

                assert source_loc < ref_code.lengths[ref_code.sort_to_orig[batch_idx]]
                edit_list.append((
                    op_idx, ref_code.raw_index(batch_idx, source_loc),
                    last_token))
            stopped = False
            try:
                next(dest_iter)
            except StopIteration:
                stopped = True
            assert stopped

            # Op = </s>, emb location and last token are irrelevant
            # edit_list.append((1, None, None))
            edit_lists.append(edit_list)

        rnn_inputs = lists_to_packed_sequence(
            [lst[:-1] for lst in edit_lists], (3,), torch.LongTensor,
            lambda op_emb_pos_last_token, _, out:
            out.copy_(torch.LongTensor([*op_emb_pos_last_token])))
        rnn_outputs = lists_to_packed_sequence(
            [lst[1:] for lst in edit_lists], (1,), torch.LongTensor,
            lambda op_emb_pos_last_token, _, out:
            out.copy_(torch.LongTensor([op_emb_pos_last_token[0]])))

        io_embed_indices = torch.LongTensor([
            expanded_idx
            for b in rnn_inputs.ps.batch_sizes
            for orig_idx in rnn_inputs.orig_to_sort[:b]
            for expanded_idx in range(orig_idx * 5, orig_idx * 5 + 5)
        ])

        return PackedDecoderData(rnn_inputs, rnn_outputs, io_embed_indices,
                                 ref_code)

    def interleave_grids_events(self, batch):
        events_lists = [
            test['trace'].events
            for item in batch for test in item.ref_example.input_tests
        ]
        result = []
        for events_list in events_lists:
            get_from_events = []
            last_timestep = None
            for ev in events_list:
                if last_timestep != ev.timestep:
                    get_from_events.append(0)
                    last_timestep = ev.timestep
                get_from_events.append(1)
            # TODO: Devise better way to test if an event is an action
            if ev.cond_span is None and ev.success:
                # Trace ends with a grid, if last event is action and it is
                # successful
                get_from_events.append(0)
            result.append(get_from_events)
        return result

    def prepare_traces_grids(self, batch):
        grids_lists = [
            test['trace'].grids
            for item in batch for test in item.ref_example.input_tests
        ]

        last_grids = [set() for _ in grids_lists]

        def fill(grid, batch_idx, out):
            if isinstance(grid, dict):
                last_grid = last_grids[batch_idx]
                assert last_grid.isdisjoint(grid['plus'])
                assert last_grid >= grid['minus']
                last_grid.update(grid['plus'])
                last_grid.difference_update(grid['minus'])
            else:
                last_grid = last_grids[batch_idx] = set(grid)
            out.zero_()
            out.view(-1)[list(last_grid)] = 1

        ref_trace_grids = lists_to_packed_sequence(grids_lists, (15, 18, 18),
                                                   torch.FloatTensor, fill)
        return ref_trace_grids

    # def get_spans(self, batch, ref_code):
    #     spans = []
    #     for item in batch:
    #         spans_for_item = []
    #         for test in item.ref_example.input_tests:
    #             spans_for_trace = []
    #             for event in test['trace'].events:
    #                 spans_for_trace.append(
    #                     (event.timestep, event.span, event.cond_span))
    #             spans_for_item.append(spans_for_trace)
    #         spans.append(spans_for_item)
    #     return Spans(spans)

    # def prepare_traces_events(self, batch, ref_code):
    #     # Split into action and cond events
    #     all_action_events = []
    #     all_cond_events = []
    #     interleave_indices = []
    #     for item in batch:
    #         for test in item.ref_example.input_tests:
    #             action_events, cond_events, interleave = [], [], []
    #             for event in test['trace'].events:
    #                 # TODO: Devise better way to test if an event is an action
    #                 if event.cond_span is None:
    #                     action_events.append(event)
    #                     interleave.append(1)
    #                 else:
    #                     cond_events.append(event)
    #                     interleave.append(0)
    #             all_action_events.append(action_events)
    #             all_cond_events.append(cond_events)
    #             interleave_indices.append(interleave)
    #
    #     packed_action_events = lists_to_packed_sequence(
    #         all_action_events,
    #         [2],
    #         torch.LongTensor,
    #         lambda ev, batch_idx, out: out.copy_(torch.LongTensor([
    #             # {'if': 0, 'ifElse': 1, 'while': 2, 'repeat': 3}[ev.type],
    #             ev.span[0], ev.success])))
    #     action_code_indices = None
    #     if ref_code:
    #         action_code_indices = Variable(torch.LongTensor(
    #             ref_code.raw_index(
    #                 # TODO: Don't hardcode 5.
    #                 # TODO: May need to work with code replicated 5 times.
    #                 packed_action_events.orig_batch_indices() // 5,
    #                 packed_action_events.ps.data.data[:, 0].numpy())))
    #
    #     packed_cond_events = lists_to_packed_sequence(
    #         all_cond_events,
    #         [6],
    #         torch.LongTensor,
    #         lambda ev, batch_idx, out: out.copy_(
    #             torch.LongTensor([
    #                 ev.span[0], ev.span[1],
    #                 ev.cond_span[0], ev.cond_span[1],
    #                 int(ev.cond_value) if isinstance(ev.cond_value, (bool, np.bool))
    #                 else int(ev.cond_value + 2),
    #                 int(ev.success)])))
    #     cond_code_indices = None
    #     if ref_code:
    #         cond_code_indices = Variable(torch.LongTensor(
    #             ref_code.raw_index(
    #                 # TODO: Don't hardcode 5.
    #                 np.expand_dims(
    #                     packed_cond_events.orig_batch_indices() // 5,
    #                     axis=1),
    #                 packed_cond_events.ps.data.data[:, :4].numpy())))
    #
    #     return PackedTrace(
    #         packed_action_events, action_code_indices, packed_cond_events,
    #         cond_code_indices, interleave_indices)
