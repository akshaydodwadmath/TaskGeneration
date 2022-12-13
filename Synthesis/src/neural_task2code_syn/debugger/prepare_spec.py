import collections
import itertools
import operator

import numpy as np
import torch
from src.utils.code_linearizer import traverse_pre_order
from torch.autograd import Variable


class PackedSequencePlus(collections.namedtuple('PackedSequencePlus',
                                                ['ps', 'lengths', 'sort_to_orig',
                                                 'orig_to_sort'])):
    def __new__(cls, ps, lengths, sort_to_orig, orig_to_sort):
        assert isinstance(ps.batch_sizes, (torch.LongTensor, torch.Tensor))
        sort_to_orig = np.array(sort_to_orig)
        orig_to_sort = np.array(orig_to_sort)
        self = super(PackedSequencePlus, cls).__new__(cls,
                                                      ps, lengths, sort_to_orig,
                                                      orig_to_sort)
        self.cum_batch_sizes = np.cumsum([0] + list(self.ps.batch_sizes[:-1]))
        return self

    def apply(self, fn):
        return PackedSequencePlus(
            torch.nn.utils.rnn.PackedSequence(
                fn(self.ps.data), self.ps.batch_sizes), self.lengths,
            self.sort_to_orig,
            self.orig_to_sort)

    def with_new_ps(self, ps):
        return PackedSequencePlus(ps, self.lengths, self.sort_to_orig,
                                  self.orig_to_sort)

    def pad(self, batch_first, others_to_unsort=(), padding_value=0.0):
        padded, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            self.ps, batch_first=batch_first, padding_value=padding_value)
        indices = torch.from_numpy(self.sort_to_orig)
        if self.ps.data.is_cuda:
            indices = indices.cuda()
        results = padded[indices], [seq_lengths[i] for i in self.sort_to_orig]
        return results + tuple(t[self.sort_to_orig] for t in others_to_unsort)

    def cuda(self, non_blocking=False):
        if self.ps.data.is_cuda:
            return self
        return self.apply(lambda d: d.cuda(non_blocking=non_blocking))

    def raw_index(self, orig_batch_idx, seq_idx):
        assert np.all(np.array(seq_idx) < len(self.cum_batch_sizes))
        assert np.all(np.array(orig_batch_idx) < len(self.sort_to_orig))
        result = np.take(self.cum_batch_sizes, seq_idx) + np.take(
            self.sort_to_orig, orig_batch_idx)
        assert np.all(result < len(self.ps.data))
        return result

    def select(self, orig_batch_idx, seq_idx):
        raw = self.raw_index(orig_batch_idx, seq_idx).tolist()
        return self.ps.data[raw]

    def orig_index(self, raw_idx):
        seq_idx = np.searchsorted(
            self.cum_batch_sizes, raw_idx, side='right') - 1
        batch_idx = raw_idx - self.cum_batch_sizes[seq_idx]
        orig_batch_idx = self.sort_to_orig[batch_idx]
        return orig_batch_idx, seq_idx

    def orig_batch_indices(self):
        result = []
        for bs in self.ps.batch_sizes:
            result.extend(self.orig_to_sort[:bs])
        return np.array(result)

    def orig_lengths(self):
        for sort_idx in self.sort_to_orig:
            yield self.lengths[sort_idx]

    def expand(self, k):
        # Conceptually, this function does the following:
        #   Input: d1 x ...
        #   Output: d1 * k x ... where
        #     out[0] = out[1] = ... out[k],
        #     out[k + 0] = out[k + 1] = ... out[k + k],
        #   and so on.
        v = self.ps.data
        ps_data = v.unsqueeze(1).repeat(1, k, *(
                [1] * (v.dim() - 1))).view(-1, *v.shape[1:])
        batch_sizes = (np.array(self.ps.batch_sizes) * k).tolist()
        lengths = np.repeat(self.lengths, k).tolist()
        sort_to_orig = [
            exp_i for i in self.sort_to_orig for exp_i in range(i * k, i * k + k)
        ]
        orig_to_sort = [
            exp_i for i in self.orig_to_sort for exp_i in range(i * k, i * k + k)
        ]

        batch_sizes = torch.tensor(batch_sizes, dtype=torch.long)
        return PackedSequencePlus(
            torch.nn.utils.rnn.PackedSequence(ps_data, batch_sizes),
            lengths, sort_to_orig, orig_to_sort)

    def cpu(self):
        if not self.ps.data.is_cuda:
            return self
        return self.apply(lambda d: d.cpu())

    def cat_with_list(self, other):
        """
        Concatenate this and the given list of sequences. These must have identical shapes and lengths for each item
        """
        assert list(self.orig_lengths()) == [x.shape[0] for x in
                                             other], "lengths of sequences do not match"
        assert len({x.shape[1:] for x in
                    other}) == 1, "input data's embedding dimension is non-homogenous"
        assert other[0].shape[2:] == self.ps.data.shape[
                                     2:], "should match at indices other than 1"

        new_data = Variable(torch.zeros([self.ps.data.shape[0], *other[0].shape[1:]]))
        if self.ps.data.is_cuda:
            new_data = new_data.cuda()

        for batch_idx, tens in enumerate(other):
            indices = Variable(torch.LongTensor(
                self.raw_index(batch_idx, list(range(len(tens)))).tolist()))
            if self.ps.data.is_cuda:
                indices = indices.cuda()

            new_data[indices] = tens

        cat_data = torch.cat([new_data, self.ps.data], dim=1)
        cat_ps = torch.nn.utils.rnn.PackedSequence(
            cat_data,
            self.ps.batch_sizes
        )
        return self.with_new_ps(cat_ps)

    def cat_with_item(self, items):
        """
        Concatenate this sequnce with each item, where the item is spread across the sequence.

        Works for all dimensions of data, it just needs to line up with the outputs correctly
        """
        assert len(items) == len(list(self.orig_lengths()))
        return self.cat_with_list([
            item.unsqueeze(0).repeat(length, *([1] * len(item.shape)))
            for item, length in zip(items, self.orig_lengths())
        ])


def batch_bounds_for_packing(lengths):
    '''Returns how many items in batch have length >= i at step i.

    Examples:
      [5] -> [1, 1, 1, 1, 1]
      [5, 5] -> [2, 2, 2, 2, 2]
      [5, 3] -> [2, 2, 2, 1, 1]
      [5, 4, 1, 1] -> [4, 2, 2, 2, 1]
    '''

    last_length = 0
    count = len(lengths)
    result = []
    for i, (length, group) in enumerate(itertools.groupby(reversed(lengths))):
        # TODO: Check that things don't blow up when some lengths are 0
        if i > 0 and length <= last_length:
            raise ValueError('lengths must be decreasing and positive')
        result.extend([count] * (length - last_length))
        count -= sum(1 for _ in group)
        last_length = length
    return result


def lengths(lsts):
    return [len(lst) for lst in lsts]


def numpy_to_tensor(arr, cuda, volatile):
    t = torch.LongTensor(arr)
    if cuda:
        t = t.cuda()
    return Variable(t, volatile=volatile)


def lists_to_numpy(lsts, stoi, default_value):
    max_length = max(lengths(lsts))
    data = np.full((len(lsts), max_length), default_value, dtype=np.int64)
    for i, lst in enumerate(lsts):
        for j, element in enumerate(lst):
            data[i, j] = stoi(element)
    return data


def sort_lists_by_length(lists):
    # lists_sorted: lists sorted by length of each element, descending
    # orig_to_sort: tuple of integers, satisfies the following:
    #   tuple(lists[i] for i in orig_to_sort) == lists_sorted
    #   lists[orig_to_sort[sort_idx]] == lists_sorted[sort_idx]
    orig_to_sort, lists_sorted = zip(*sorted(
        enumerate(lists), key=lambda x: len(x[1]), reverse=True))
    # sort_to_orig: list of integers, satisfies the following:
    #   [lists_sorted[i] for i in sort_to_orig] == lists
    #   lists_sorted[sort_to_orig[orig_idx]] == lists[orig_idx]
    sort_to_orig = [
        x[0] for x in sorted(
            enumerate(orig_to_sort), key=operator.itemgetter(1))
    ]

    return lists_sorted, sort_to_orig, orig_to_sort


def lists_to_packed_sequence_translated(lists, vocab, cuda, volatile):
    lines = [
        code +
        [vocab['END']]
        for code in lists]

    lists_sorted, sort_to_orig, orig_to_sort = sort_lists_by_length(lines)

    lens = [len(line) for line in lines]
    max_len = max(lens)

    tgt_lines = [line[0:] + [vocab['PAD']] * (max_len - len(line)) for line in
                 lines]

    v = torch.tensor(tgt_lines).long()
    lens = lengths(lists_sorted)
    lens = torch.tensor(lens, dtype=torch.long)
    return PackedSequencePlus(
        torch.nn.utils.rnn.pack_padded_sequence(
            v, lens, batch_first=True),
        lens,
        sort_to_orig,
        orig_to_sort)


def lists_to_packed_sequence(lists, vocab, cuda, volatile):
    # # Note, we are not using pack_sequence here, because it takes a list of Variables which we want to avoid since it
    # # may cause memory fragmentation.
    #
    # # lists_sorted: lists sorted by length of each element, descending
    # # orig_to_sort: tuple of integers, satisfies the following:
    # #   tuple(lists[i] for i in orig_to_sort) == lists_sorted
    # orig_to_sort, lists_sorted = zip(
    #     *sorted(enumerate(lists), key=lambda x: len(x[1]), reverse=True))
    # # sort_to_orig: list of integers, satisfies the following:
    # #   [lists_sorted[i] for i in sort_to_orig] == lists
    # sort_to_orig = [x[0] for x in sorted(
    #     enumerate(orig_to_sort), key=operator.itemgetter(1))]

    def translate_code(code):
        traversed_code_tokens = np.array(
            [vocab[t] for t in traverse_pre_order(code, teacher_code=True)])
        traversed_code_tokens = np.append(traversed_code_tokens, 0)
        return traversed_code_tokens

    lines = [
        translate_code(code).tolist() +
        [vocab['END']]
        for code in lists]

    lists_sorted, sort_to_orig, orig_to_sort = sort_lists_by_length(lines)

    lens = [len(line) for line in lines]
    max_len = max(lens)

    tgt_lines = [line[0:] + [vocab['PAD']] * (max_len - len(line)) for line in
                 lines]

    v = torch.tensor(tgt_lines).long()
    lens = lengths(lists_sorted)
    lens = torch.tensor(lens, dtype=torch.long)
    return PackedSequencePlus(
        torch.nn.utils.rnn.pack_padded_sequence(
            v, lens, batch_first=True),
        lens,
        sort_to_orig,
        orig_to_sort)
