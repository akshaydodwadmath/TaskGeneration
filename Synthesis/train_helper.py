import itertools
import torch
import torch.autograd as autograd
from torch.autograd import Variable

def do_supervised_minibatch(model,
                            # Source
                            inp_grids, out_grids,
                            # Target
                            in_tgt_seq, in_tgt_seq_list, out_tgt_seq,
                            # Criterion
                            criterion):

    # Get the log probability of each token in the ground truth sequence of tokens.
    decoder_logit, _ = model(inp_grids, out_grids, in_tgt_seq, in_tgt_seq_list)
    
    nb_predictions = torch.numel(out_tgt_seq.data)
    
    ## out_tgt_seq will be of size batch size x max seq length, non max sequences will be padded
    ## decoder_logit will be of size (out_tgt_seq size, vocab size(52))
    ## cross entropy loss requires unnormalized prediction scores of all tokens, output class

    # criterion is a weighted CrossEntropyLoss. The weights are used to not penalize
    # the padding prediction used to make the batch of the appropriate size.
    loss = criterion(
        decoder_logit.contiguous().view(nb_predictions, decoder_logit.size(2)),
        out_tgt_seq.view(nb_predictions)
    )

    # Do the backward pass over the loss
    loss.backward()
    
    print('loss', loss.item())

    # Return the value of the loss over the minibatch for monitoring
    return loss.item()
    #return loss.data[0]
