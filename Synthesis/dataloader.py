# External imports
import json
import random
import torch
import os

from tqdm import tqdm
from torch.autograd import Variable
from karel.world import World
from itertools import chain


IMG_FEAT = 5184
IMG_DIM = 18
IMG_SIZE = torch.Size((16, IMG_DIM, IMG_DIM))

actions = [
    'move',
    'turnLeft',
    'turnRight',
    'pickMarker',
    'putMarker',
]

commands = ['REPEAT',
            'WHILE',
            'IF',
            'IFELSE',
            'ELSE',
            ]

def grid_desc_to_tensor(grid_desc):
    grid = torch.Tensor(IMG_FEAT).fill_(0)
    grid.index_fill_(0, grid_desc.long(), 1)
    grid = grid.view(IMG_SIZE)
    return grid
    
def translate(seq,
              vocab):
    return [vocab[str(elt)] for elt in seq]
    
def load_input_file(path_to_dataset, path_to_vocab):
    '''
    path_to_dataset: File containing the data
    path_to_vocab: File containing the vocabulary
    '''
    tgt_tkn2idx = {
        '<pad>': 0,
    }
    next_id = 1
    with open(path_to_vocab, 'r') as vocab_file:
        for line in vocab_file.readlines():
            tgt_tkn2idx[line.strip()] = next_id
            next_id += 1
    tgt_idx2tkn = {}
    for tkn, idx in tgt_tkn2idx.items():
        tgt_idx2tkn[idx] = tkn

    vocab = {"idx2tkn": tgt_idx2tkn,
             "tkn2idx": tgt_tkn2idx}

    path_to_ds_cache = path_to_dataset.replace('.json', '.thdump')
    if os.path.exists(path_to_ds_cache):
        dataset = torch.load(path_to_ds_cache)
    else:
        with open(path_to_dataset, 'r') as dataset_file:
            srcs = []
            tgts = []
            fVectors = []
            for sample_str in tqdm(dataset_file.readlines()):
                sample_data = json.loads(sample_str)

                # Get the target program
                tgt_program_tkn = sample_data['Code']

                tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
                
                src_feature_index = sample_data['FeatureVectorIndex']

                srcs.append(src_feature_index)
                tgts.append(tgt_program_idces)
                fVectors.append(sample_data['FeatureVector'])
        
        dataset = {"sources": srcs,
                   "targets": tgts,
                   "featureVectors": fVectors}
        torch.save(dataset, path_to_ds_cache)
    return dataset, vocab, max(dataset["sources"])+1    

#TODO: undestand this
def shuffle_dataset(dataset, batch_size, randomize=True):
    '''
    We are going to group together samples that have a similar length, to speed up training
    batch_size is passed so that we can align the groups
    '''
    pairs = list(zip(dataset["sources"], dataset["targets"]))
    bucket_fun = lambda x: len(x[1]) / 5
    pairs.sort(key=bucket_fun, reverse=True)
    grouped_pairs = [pairs[pos: pos + batch_size]
                     for pos in range(0,len(pairs), batch_size)]
    if randomize:
        to_shuffle = grouped_pairs[:-1]
        random.shuffle(to_shuffle)
        grouped_pairs[:-1] = to_shuffle
    pairs = chain.from_iterable(grouped_pairs)
    in_seqs, out_seqs = zip(*pairs)
    return {
        "sources": in_seqs,
        "targets": out_seqs
    }

def get_minibatch(dataset, sp_idx, batch_size,
                  start_idx, end_idx, pad_idx,
                  shuffle=True, volatile_vars=False):
    """Prepare minibatch."""

    # Prepare the grids
    srcs  = dataset["sources"][sp_idx:sp_idx+batch_size]

    # Prepare the target sequences
    targets = dataset["targets"][sp_idx:sp_idx+batch_size]
    
    srcs_fVector = dataset["featureVectors"][sp_idx:sp_idx+batch_size]

    lines = [
        [start_idx] + line for line in targets
    ]
    lens = [len(line) for line in lines]
    max_len = max(lens)
    
    
    # Drop the last element, it should be the <end> symbol for all of them
    # padding for all of them
    input_lines = [
        line[:max_len-1] + [pad_idx] * (max_len - len(line[:max_len-1])-1) for line in lines
    ]

    # Drop the first element, should always be the <start> symbol. This makes
    # everything shifted by one compared to the input_lines
    output_lines = [
        line[1:] + [pad_idx] * (max_len - len(line)) for line in lines
    ]

    in_src_seq = Variable(torch.LongTensor(srcs), volatile=volatile_vars)
    tgt_inp_sequences = Variable(torch.LongTensor(input_lines), volatile=volatile_vars)
    out_tgt_seq = Variable(torch.LongTensor(output_lines), volatile=volatile_vars)

    return tgt_inp_sequences, in_src_seq, out_tgt_seq, srcs,targets,srcs_fVector
