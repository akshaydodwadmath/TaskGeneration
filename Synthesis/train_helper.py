import itertools
import torch
import torch.autograd as autograd
from torch.autograd import Variable


def do_supervised_minibatch(model, tgt_inp_sequences, in_src_seq, out_tgt_seq, criterion, weight_lambda):

    # Get the log probability of each token in the ground truth sequence of tokens.
    
    decoder_logit, probs, _ = model(tgt_inp_sequences, in_src_seq, out_tgt_seq)
   
    probs_avg = torch.mean(torch.add(probs,1e-6), 0, True)
    loss_entropy = -torch.matmul(probs_avg, torch.log(probs_avg).reshape(-1,1))
    nb_predictions = torch.numel(out_tgt_seq.data)
    ## out_tgt_seq will be of size batch size x max seq length, non max sequences will be padded
    ## decoder_logit will be of size (out_tgt_seq size, vocab size(52))
    ## cross entropy loss requires unnormalized prediction scores of all tokens, output class

    # criterion is a weighted CrossEntropyLoss. The weights are used to not penalize
    # the padding prediction used to make the batch of the appropriate size.
    loss_train = criterion(
        decoder_logit.contiguous().view(nb_predictions, decoder_logit.size(2)),
        out_tgt_seq.view(nb_predictions)
    )
    #print("decoder_logit", torch.argmax(decoder_logit, dim=2))
    loss = loss_train - weight_lambda * loss_entropy
    # Do the backward pass over the loss
    loss.backward()
    
    print('loss', loss.item())

    # Return the value of the loss over the minibatch for monitoring
    return loss.item(), loss_train.item(), loss_entropy.item()
    #return loss.data[0]
