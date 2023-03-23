import itertools
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import random

def do_supervised_minibatch(model, tgt_inp_sequences, in_src_seq, tgt_seq_list, out_tgt_seq, nb_actions_seq, criterion, weight_lambda):

    # Get the log probability of each token in the ground truth sequence of tokens.
    
    decoder_logit, probs, _ = model(tgt_inp_sequences, in_src_seq, tgt_seq_list, out_tgt_seq, nb_actions_seq)
   
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

def do_rl_minibatch(model,
                    # Source
                    in_src_seq,
                    # Target
                    nb_actions_seq,
                    envs,
                    # Config
                    tgt_start_idx, tgt_end_idx, max_len,
                    n_domains,
                    nb_rollouts,
                    entropy_weight):

    if in_src_seq.is_cuda:
        use_cuda = True
            
    tgt_encoder_vector = torch.Tensor(len(in_src_seq), n_domains).fill_(0)
    index = torch.tensor(random.sample(range(n_domains), len(in_src_seq)))
    
    if use_cuda:
        tgt_encoder_vector, index = tgt_encoder_vector.cuda(), index.cuda()
    for i, ind in enumerate(index):
        tgt_encoder_vector[i].index_fill_(0, ind, 1)
    
    # Samples `nb_rollouts` samples from the decoding model.
    rolls = model.sample_model(in_src_seq,tgt_encoder_vector,
                               nb_actions_seq,
                            tgt_start_idx, tgt_end_idx, max_len,
                            nb_rollouts)
    for roll, env in zip(rolls, envs):
        # Assign the rewards for each sample
        roll.assign_rewards(env, [], entropy_weight)

    # Evaluate the performance on the minibatch
    batch_reward = sum(roll.dep_reward for roll in rolls)
    # Get all variables and all gradients from all the rolls
    variables, grad_variables = zip(*batch_rolls_reinforce(rolls))

    # For each of the sampling probability, we know their gradients.
    # See https://arxiv.org/abs/1506.05254 for what we are doing,
    # simply using the probability of the choice made, times the reward of all successors.
    autograd.backward(variables, grad_variables)
    

    # Return the value of the loss/reward over the minibatch for convergence
    # monitoring.
    return batch_reward

def batch_rolls_reinforce(rolls):
    for roll in rolls:
        for var, grad in roll.yield_var_and_grad():
            if grad is None:
                assert var.requires_grad is False
            else:
                yield var, grad
