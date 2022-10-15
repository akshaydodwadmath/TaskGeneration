
# External imports
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from dataloader import IMG_SIZE
from misc.beam import Beam

### A module to ensure convolution happens in the correct manner and the required dimensions are obtained 
### after convolution. Example:
# inp_grids torch.Size([16, 5, 16, 18, 18])
# x_feat_shape torch.Size([16, 18, 18])
# flat_x_shape (-1, 16, 18, 18)
# flat_x torch.Size([80, 16, 18, 18]) : x is input
# flat_y torch.Size([80, 32, 18, 18]) : y is output
# y_feat_shape torch.Size([32, 18, 18])
# y_shape torch.Size([16, 5, 32, 18, 18])
class MapModule(nn.Module):
    '''
    Takes as argument a module `elt_module` that as a signature:
    B1 x I1 x I2 x I3 x ... -> B x O1 x O2 x O3 x ...
    This becomes a module with signature:
    B1 x B2 x B3 ... X I1 x I2 x I3 -> B1 x B2 x B3 x ... X O1 x O2 x O3
    '''
    def __init__(self, elt_module, nb_mod_dim):
        super(MapModule, self).__init__()
        self.elt_module = elt_module
        self.nb_mod_dim = nb_mod_dim

    def forward(self, x):
        x_batch_shape = x.size()[:-self.nb_mod_dim]
        x_feat_shape = x.size()[-self.nb_mod_dim:]

        flat_x_shape = (-1, ) + x_feat_shape
        flat_x = x.contiguous().view(flat_x_shape)
        flat_y = self.elt_module(flat_x)

        y_feat_shape = flat_y.size()[1:]
        y_shape = x_batch_shape + y_feat_shape
        y = flat_y.view(y_shape)

        return y
        
        
class MultiIOProgramDecoder(nn.Module):
    '''
    This LSTM based decoder offers two methods to obtain programs,
    based on a batch of embeddings of the IO grids.

    - `beam_sample` will return the `top_k` best programs, according to
      a beam search using `beam_size` rays.
      Outputs are under the form of tuples
      (Variable with the log proba of the sequence, sequence (as a list) )
    - `sample_model` will sample directly from the probability distribution
      defined by the model.
      Outputs are under the forms of `Rolls` objects. The expected use is to
      assign rewards to the trajectory (using the `Rolls.assign_rewards` function)
      and then use the `yield_*` functions to get them.
    '''
    def __init__(self, vocab_size, embedding_dim,
                 io_emb_size, lstm_hidden_size, nb_layers,
                 learn_syntax):
        super(MultiIOProgramDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.io_emb_size = io_emb_size
        self.lstm_input_size = io_emb_size + embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.nb_layers = nb_layers
        self.syntax_checker = None
        self.learned_syntax_checker = None

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_dim
        )
        self.rnn = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            self.nb_layers,
        )

        self.initial_h = nn.Parameter(torch.Tensor(self.nb_layers, 1, 1, self.lstm_hidden_size))
        self.initial_c = nn.Parameter(torch.Tensor(self.nb_layers, 1, 1, self.lstm_hidden_size))

        self.out2token = MapModule(nn.Linear(self.lstm_hidden_size, self.vocab_size), 1)
        #TODO
        if learn_syntax:
            self.learned_syntax_checker = SyntaxLSTM(self.vocab_size, self.embedding_dim,
                                                     self.lstm_hidden_size, self.nb_layers)

        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.out2token.elt_module.bias.data.fill_(0)
        self.initial_h.data.uniform_(-initrange, initrange)
        self.initial_c.data.uniform_(-initrange, initrange)
        
    
    def forward(self, tgt_inp_sequences, io_embeddings,
                list_inp_sequences,
                initial_state=None,
                grammar_state=None):
        '''
        tgt_inp_sequences: batch_size x seq_len(tensor length=64)
        io_embeddings: batch_size x nb_ios x io_emb_size
        '''
        # unsqueeze:  dimension of size one inserted at the specified position.
        # squeeze: dimensions of input of size 1 removed.
        # expand: singleton dimensions expanded to a larger size.
        # permute:  dimensions permuted.
        # torch.cat dim: dimensions along which the tensors are concatenated
        # TODO: understand seq_len again
        batch_size, seq_len = tgt_inp_sequences.size()
        _, nb_ios, _ = io_embeddings.size()
     #  print('tgt_inp_sequences',tgt_inp_sequences)
     #   print('tgt_inp_sequences.long()', tgt_inp_sequences.long())
        seq_emb = self.embedding(tgt_inp_sequences).permute(1, 0, 2).contiguous()
        # seq_emb: seq_len x batch_size x embedding_dim
        per_io_seq_emb = seq_emb.unsqueeze(2).expand(seq_len, batch_size, nb_ios, self.embedding_dim)
        # per_io_seq_emb: seq_len x batch_size x nb_ios x embedding_dim

        lstm_cell_size = torch.Size((self.nb_layers, batch_size, nb_ios, self.lstm_hidden_size))
        if initial_state is None:
            initial_state = (
                self.initial_h.expand(lstm_cell_size).contiguous(),
                self.initial_c.expand(lstm_cell_size).contiguous()
            )
        

        # Add the IO embedding to each of the input
        io_embeddings = io_embeddings.unsqueeze(0)
        io_embeddings = io_embeddings.expand(seq_len, batch_size, nb_ios, self.io_emb_size)


        # Forms the input correctly
        dec_input = torch.cat([per_io_seq_emb, io_embeddings], 3)
        # dec_input: seq_len x batch_size x nb_ios x lstm_input_size

        # Flatten across batch x nb_ios
        dec_input = dec_input.view(seq_len, batch_size*nb_ios, self.lstm_input_size)
        # dec_input: seq_len x batch_size*nb_ios x lstm_input_size
        initial_state = (
            initial_state[0].view(self.nb_layers, batch_size*nb_ios, self.lstm_hidden_size),
            initial_state[1].view(self.nb_layers, batch_size*nb_ios, self.lstm_hidden_size)
        )
        # initial_state: 2-Tuple of (nb_layers x batch_size*nb_ios x embedding_dim)

        # Pass through the LSTM
        dec_out, dec_lstm_state = self.rnn(dec_input.contiguous(), initial_state)
        # dec_out: seq_len x batch_size*nb_ios x lstm_hidden_size
        # dec_lstm_state: 2-Tuple of (nb_layers x batch_size*nb_ios x embedding_dim)

        # Reshape the output:
        dec_out = dec_out.view(1, seq_len, batch_size, nb_ios, self.lstm_hidden_size)
        # XXX there is an extremely weird bug in pytorch when doing the max
        # over dim = 2 so we introduce a dummy dimension to avoid it, so
        # that we can operate on the third dimension
        pool_out, _ = dec_out.max(3)
        pool_out = pool_out.squeeze(3).squeeze(0)
        # pool_out: seq_len x batch_size x lstm_hidden_size

        # Flatten for decoding
        decoder_logit = self.out2token(pool_out)
        # decoder_logit: seq_len x batch_size x out_voc_size
        decoder_logit = decoder_logit.permute(1, 0, 2)
        # decoder_logit: batch_size x seq_len x out_voc_size

        # Also reorganise the state
        dec_lstm_state = (
            dec_lstm_state[0].view(self.nb_layers, batch_size, nb_ios, self.lstm_hidden_size),
            dec_lstm_state[1].view(self.nb_layers, batch_size, nb_ios, self.lstm_hidden_size)
        )
        syntax_mask = None
        
        #TODO if self.syntax_checker is not None:
        
        return decoder_logit, dec_lstm_state, grammar_state, syntax_mask
    
       #TODO understand beam search, currently only used for model evaluation
    
    def beam_sample(self, io_embeddings,
                    tgt_start, tgt_end, max_len,
                    beam_size, top_k, vol):

        '''
        io_embeddings: batch_size x nb_ios x io_emb_size
        All the rest are ints
        vol is a boolean indicating whether created Variables should be volatile
        '''
        batch_size, nb_ios, io_emb_size = io_embeddings.size()
        use_cuda = io_embeddings.is_cuda
        tt = torch.cuda if use_cuda else torch
        force_beamcpu = True

        beams = [Beam(beam_size, top_k, tgt_start, tgt_end, use_cuda and not force_beamcpu)
                 for _ in range(batch_size)]

        lsm = nn.LogSoftmax(dim=1)

        # We will make it a batch size of beam_size
        batch_state = None  # First one is the learned default state
        batch_grammar_state = None
        batch_inputs = Variable(tt.LongTensor(batch_size, 1).fill_(tgt_start), volatile=vol)
        batch_list_inputs = [[tgt_start]]*batch_size
        batch_io_embeddings = io_embeddings
        batch_idx = Variable(torch.arange(0, batch_size, 1).long(), volatile=vol)
        if use_cuda:
            batch_idx = batch_idx.cuda()
        beams_per_sp = [1 for _ in range(batch_size)]

        for stp in range(max_len):
            # We will just do the forward of one timestep Each of the inputs
            # for a beam appears as a different sample in the batch

           # if batch_state is not None:
               # print('in_tgt_seq_list', (batch_state.size()))
           # print('batch_inputs', batch_inputs)
            dec_outs, dec_state, \
            batch_grammar_state, _ = self.forward(batch_inputs,
                                                  batch_io_embeddings,
                                                  batch_list_inputs,
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
                if self.syntax_checker is not None:
                    for idx in sp_parent_idxs.data:
                        new_batch_checker.append(copy.copy(batch_grammar_state[idx]))
                sp_next_batch_idxs = Variable(tt.LongTensor(sp_curr_beam_size).fill_(i),
                                              volatile=vol)
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

            assert(sp_from_idx == lpb_to_use.size(0))  # have we gone over all the things?
            if len(new_inputs)==0:
                # All of our beams are done
                break
            batch_inputs = torch.cat(new_inputs, 0)
            batch_idx = torch.cat(new_batch_idx, 0)
            parent_idxs = torch.cat(new_parent_idxs, 0)
            batch_list_inputs = new_batch_list_inputs

            batch_state = (
                dec_state[0].index_select(1, parent_idxs.type(torch.int64)),
                dec_state[1].index_select(1, parent_idxs.type(torch.int64))
                )
            if self.syntax_checker is not None:
                batch_grammar_state = new_batch_checker
            elif self.learned_syntax_checker is not None:
                batch_grammar_state = (
                    batch_grammar_state[0].index_select(1, parent_idxs),
                    batch_grammar_state[1].index_select(1, parent_idxs)
                )
            batch_io_embeddings = io_embeddings.index_select(0, batch_idx)
            beams_per_sp = new_beams_per_sp
            assert(len(beams_per_sp)==len(beams))
        sampled = []
        for i, beamState in enumerate(beams):
            sampled.append(beamState.get_sampled())
        return sampled

class GridEncoder(nn.Module):
    def __init__(self, kernel_size, conv_stack, fc_stack):
        '''
        kernel_size: width of the kernels
        conv_stack: Number of channels at each point of the convolutional part of
                    the network (includes the input)
        fc_stack: number of channels in the fully connected part of the network
        '''
        super(GridEncoder, self).__init__()
        self.conv_layers = []
        for i in range(1, len(conv_stack)):
            if conv_stack[i-1] != conv_stack[i]:
                block = nn.Sequential(
                    ResBlock(kernel_size, conv_stack[i-1]),
                    nn.Conv2d(conv_stack[i-1], conv_stack[i],
                              kernel_size=kernel_size, padding=(kernel_size-1)//2 ),
                    nn.ReLU(inplace=True)
                )
            else:
                block = ResBlock(kernel_size, conv_stack[i-1])
            self.conv_layers.append(block)
            self.add_module("ConvBlock-" + str(i-1), self.conv_layers[-1])

        # We have operated so far to preserve all of the spatial dimensions so
        # we can estimate the flattened dimension.
        first_fc_dim = conv_stack[-1] * IMG_SIZE[-1]* IMG_SIZE[-2]
        adjusted_fc_stack = [first_fc_dim] + fc_stack
        self.fc_layers = []
       
        for i in range(1, len(adjusted_fc_stack)):
            self.fc_layers.append(nn.Linear(adjusted_fc_stack[i-1],
                                            adjusted_fc_stack[i]))
            self.add_module("FC-" + str(i-1), self.fc_layers[-1])
        # TODO: init?

    def forward(self, x):
        '''
        x: batch_size x channels x Height x Width
        '''
        batch_size = x.size(0)

        # Convolutional part
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten for the fully connected part
        x = x.view(batch_size, -1)
        # Fully connected part
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)

        return x

class ResBlock(nn.Module):
    def __init__(self, kernel_size, in_feats):
        '''
        kernel_size: width of the kernels
        in_feats: number of channels in inputs
        '''
        super(ResBlock, self).__init__()
        self.feat_size = in_feats
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv2 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv3 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += residual
        out = self.relu(out)

        return out
        
class IOsEncoder(nn.Module):
    def __init__(self, kernel_size, conv_stack, fc_stack):
        super(IOsEncoder, self).__init__()

        ## Do one layer of convolution before stacking

        # Deduce the size of the embedding for each grid
        initial_dim = conv_stack[0] // 2  # Because we are going to get dim from I and dim from O

        # TODO: we know that our grids are mostly sparse, and only positive.
        # That means that a different initialisation might be more appropriate.
        self.in_grid_enc = MapModule(nn.Sequential(
            nn.Conv2d(IMG_SIZE[0], initial_dim, kernel_size=kernel_size, padding=(kernel_size -1)//2),
            nn.ReLU(inplace=True)
        ), 3)
        self.out_grid_enc = MapModule(nn.Sequential(
            nn.Conv2d(IMG_SIZE[0], initial_dim,kernel_size=kernel_size, padding=(kernel_size -1)//2),
            nn.ReLU(inplace=True)
        ), 3)

        # Define the model that works on the stacking
        self.joint_enc = MapModule(nn.Sequential(
            GridEncoder(kernel_size, conv_stack, fc_stack)
        ), 3)

    def forward(self, input_grids, output_grids):
        '''
        {input, output}_grids: batch_size x nb_ios x channels x height x width
        '''
        inp_emb = self.in_grid_enc(input_grids)
        out_emb = self.out_grid_enc(output_grids)
        # {inp, out}_emb: batch_size x nb_ios x feats x height x width

        io_emb = torch.cat([inp_emb, out_emb], 2)
        # io_emb: batch_size x nb_ios x 2 * feats x height x width
        joint_emb = self.joint_enc(io_emb)
        # return joint_emb
        return joint_emb
        
class IOs2Seq(nn.Module):

    def __init__(self,
                 # IO encoder
                 kernel_size, conv_stack, fc_stack,
                 # Program Decoder
                 tgt_vocabulary_size,
                 tgt_embedding_dim,
                 decoder_lstm_hidden_size,
                 decoder_nb_lstm_layers,
                 learn_syntax):
        super(IOs2Seq, self).__init__()
        self.encoder = IOsEncoder(kernel_size, conv_stack, fc_stack)
        io_emb_size = fc_stack[-1]
        self.decoder = MultiIOProgramDecoder(tgt_vocabulary_size,
                                             tgt_embedding_dim,
                                             io_emb_size,
                                             decoder_lstm_hidden_size,
                                             decoder_nb_lstm_layers,
                                             learn_syntax)
                                             
                                             
    def forward(self, input_grids, output_grids, tgt_inp_sequences, list_inp_sequences):

        io_embedding = self.encoder(input_grids, output_grids)
        dec_outs, _, _, syntax_mask = self.decoder(tgt_inp_sequences,
                                                   io_embedding,
                                                   list_inp_sequences)
        return dec_outs, syntax_mask
    
    
    def beam_sample(self, input_grids, output_grids,
                    tgt_start, tgt_end, max_len,
                    beam_size, top_k, vol=True):
        io_embedding = self.encoder(input_grids, output_grids)

        sampled = self.decoder.beam_sample(io_embedding,
                                           tgt_start, tgt_end, max_len,
                                           beam_size, top_k, vol)
        return sampled
