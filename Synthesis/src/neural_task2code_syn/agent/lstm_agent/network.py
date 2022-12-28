import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.neural_task2code_syn.agent.lstm_agent.rolls import Rolls

IMG_DIM = 18
IMG_SIZE = torch.Size((16, IMG_DIM, IMG_DIM))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MapModule(nn.Module):
    """
    Takes as argument a module `elt_module` that as a signature:
    B1 x I1 x I2 x I3 x ... -> B x O1 x O2 x O3 x ...
    This becomes a module with signature:
    B1 x B2 x B3 ... X I1 x I2 x I3 -> B1 x B2 x B3 x ... X O1 x O2 x O3
    """

    def __init__(self, elt_module, nb_mod_dim):
        super(MapModule, self).__init__()
        self.elt_module = elt_module
        self.nb_mod_dim = nb_mod_dim

    def forward(self, x):
        x_batch_shape = x.size()[:-self.nb_mod_dim]
        x_feat_shape = x.size()[-self.nb_mod_dim:]

        flat_x_shape = (-1,) + x_feat_shape
        flat_x = x.contiguous().view(flat_x_shape)
        flat_y = self.elt_module(flat_x)

        y_feat_shape = flat_y.size()[1:]
        y_shape = x_batch_shape + y_feat_shape
        y = flat_y.view(y_shape)

        return y


class SyntaxLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_size,
                 nb_layers):
        super(SyntaxLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.lstm_input_size = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.nb_layers = nb_layers

        self.rnn = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            self.nb_layers
        )
        self.initial_h = nn.Parameter(
            torch.Tensor(self.nb_layers, 1, self.lstm_hidden_size))
        self.initial_c = nn.Parameter(
            torch.Tensor(self.nb_layers, 1, self.lstm_hidden_size))

        self.out2token = MapModule(nn.Linear(self.lstm_hidden_size, self.vocab_size), 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.out2token.elt_module.bias.data.fill_(0)
        self.initial_h.data.uniform_(-initrange, initrange)
        self.initial_c.data.uniform_(-initrange, initrange)

    def forward(self, inp_sequence_embedded, state):
        """
        inp_sequence_embedded: seq_len x batch_size x embedding_dim
        state: 2 tuple of (nb_layers x batch_size x hidden_size)
        """
        seq_len, batch_size, _ = inp_sequence_embedded.size()

        if state is None:
            lstm_state_size = torch.Size(
                (self.nb_layers, batch_size, self.lstm_hidden_size))
            state = (
                self.initial_h.expand(lstm_state_size).contiguous(),
                self.initial_c.expand(lstm_state_size).contiguous()
            )
        stx_out, state = self.rnn(inp_sequence_embedded, state)

        stx_scores = self.out2token(stx_out)
        # seq_len x batch_size x out_vocab_size

        stx_mask = -stx_scores.exp()
        # seq_len x batch_size x out_vocab_size
        stx_mask = stx_mask.permute(1, 0, 2)
        # batch_size x seq_len x out_voc_size

        return stx_mask, state


class MultiIOProgramDecoder(nn.Module):
    """
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
    """

    def __init__(self, vocab_size, embedding_dim,
                 io_emb_size, lstm_hidden_size, nb_layers):
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
        ).to(device)

        self.rnn = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            self.nb_layers,
        ).to(device)

        self.initial_h = nn.Parameter(
            torch.Tensor(self.nb_layers, 1, 1, self.lstm_hidden_size)).to(device)
        self.initial_c = nn.Parameter(
            torch.Tensor(self.nb_layers, 1, 1, self.lstm_hidden_size)).to(device)

        self.out2token = MapModule(nn.Linear(self.lstm_hidden_size, self.vocab_size),
                                   1).to(device)

        self.init_weights()

    def __getstate__(self):
        # Don't dump the syntax checker
        obj_dict = self.__dict__.copy()
        obj_dict["syntax_checker"] = None
        return obj_dict

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.out2token.elt_module.bias.data.fill_(0)
        self.initial_h.data.uniform_(-initrange, initrange)
        self.initial_c.data.uniform_(-initrange, initrange)

    def forward(self, tgt_inp_sequences, io_embeddings,
                initial_state=None,
                grammar_state=None):
        """
        tgt_inp_sequences: batch_size x seq_len
        io_embeddings: batch_size x nb_ios x io_emb_size
        """
        batch_size, seq_len = tgt_inp_sequences.shape
        _, nb_ios, _ = io_embeddings.shape
        seq_emb = self.embedding(tgt_inp_sequences).permute(1, 0, 2).contiguous()
        # seq_emb: seq_len x batch_size x embedding_dim
        per_io_seq_emb = seq_emb.unsqueeze(2).expand(seq_len, batch_size, nb_ios,
                                                     self.embedding_dim)
        # per_io_seq_emb: seq_len x batch_size x nb_ios x embedding_dim

        lstm_cell_size = torch.Size(
            (self.nb_layers, batch_size, nb_ios, self.lstm_hidden_size))
        if initial_state is None:
            initial_state = (
                self.initial_h.expand(lstm_cell_size).contiguous(),
                self.initial_c.expand(lstm_cell_size).contiguous()
            )

        # Add the IO embedding to each of the input
        io_embeddings = io_embeddings.unsqueeze(0)
        io_embeddings = io_embeddings.expand(seq_len, batch_size, nb_ios,
                                             self.io_emb_size)

        # Forms the input correctly
        dec_input = torch.cat([per_io_seq_emb, io_embeddings], 3)
        # dec_input: seq_len x batch_size x nb_ios x lstm_input_size

        # Flatten across batch x nb_ios
        dec_input = dec_input.view(seq_len, batch_size * nb_ios, self.lstm_input_size)
        # dec_input: seq_len x batch_size*nb_ios x lstm_input_size
        initial_state = (
            initial_state[0].view(self.nb_layers, batch_size * nb_ios,
                                  self.lstm_hidden_size),
            initial_state[1].view(self.nb_layers, batch_size * nb_ios,
                                  self.lstm_hidden_size)
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
            dec_lstm_state[0].view(self.nb_layers, batch_size, nb_ios,
                                   self.lstm_hidden_size),
            dec_lstm_state[1].view(self.nb_layers, batch_size, nb_ios,
                                   self.lstm_hidden_size)
        )

        return decoder_logit, dec_lstm_state, grammar_state

    def sample_model(self, io_embeddings,
                     tgt_start, tgt_end, max_len,
                     nb_samples, vol):
        """
        io_embeddings: batch_size x nb_ios x io_emb_size
        tgt_start: int -> Character indicating the start of the decoding
        tgt_end: int -> Character indicating the end of the decoding
        max_len: int -> How many samples to sample from this element of the batch
        nb_samples: int -> How many samples to collect for each of the samples
                           of the batch
        vol: boolean -> Create all Variable as volatile
        """
        # I will attempt to `detach` or create with `requires_grad=False` all
        # of the variables that won't need backpropagating through to ensure
        # that no spurious computation is done.
        if io_embeddings.is_cuda:
            # Depending on which GPU the machine has, it may be faster to do CPU.
            # For the Quadro K420, the CPU is slightly faster
            # On the Tesla K40, the GPU is twice as fast
            tt = torch.cuda
            tt.set_device(self.device)
        else:
            tt = torch

        # batch_size is going to be a changing thing, it correspond to how many
        # inputs we are passing through the decoder at once. Here, at
        # initialization, it is just the actual batch_size.
        batch_size, nb_ios, io_emb_size = io_embeddings.size()

        # rolls holds the sample output that we are going to collect for each
        # of the outputs

        # Initial proba for what is certainly sampled
        full_proba = Variable(tt.FloatTensor([1]), requires_grad=False, volatile=vol)
        rolls = [Rolls(-1, full_proba, nb_samples, -1, device=self.device) for _ in
                 range(
                     batch_size)]

        sm = nn.Softmax(dim=1)

        # Initialising the elements for the decoder
        curr_batch_size = batch_size  # Will vary as we go along in the decoder

        batch_inputs = Variable(tt.LongTensor(batch_size, 1).fill_(tgt_start),
                                volatile=vol)
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
            batch_grammar_state = self.forward(batch_inputs,
                                               batch_io_embeddings,
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
                                    requires_grad=False, volatile=vol)
            batch_list_inputs = [[inp] for inp in next_batch_inputs]
            # Which are the parents that we need to get the state for
            # (potentially multiple times the same parent)
            parents_to_continue = [parent_idx for (parent_idx, to_cont)
                                   in zip(parent, to_continue_mask) if to_cont]
            parent = Variable(tt.LongTensor(parents_to_continue), requires_grad=False,
                              volatile=vol)

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


class ResBlock(nn.Module):
    def __init__(self, kernel_size, in_feats):
        """
        kernel_size: width of the kernels
        in_feats: number of channels in inputs
        """
        super(ResBlock, self).__init__()
        self.feat_size = in_feats
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding).to(device)
        self.conv2 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding).to(device)
        self.conv3 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)

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


class GridEncoder(nn.Module):
    def __init__(self, kernel_size, conv_stack, fc_stack):
        """
        kernel_size: width of the kernels
        conv_stack: Number of channels at each point of the convolutional part of
                    the network (includes the input)
        fc_stack: number of channels in the fully connected part of the network
        """
        super(GridEncoder, self).__init__()
        self.conv_layers = []
        for i in range(1, len(conv_stack)):
            if conv_stack[i - 1] != conv_stack[i]:
                block = nn.Sequential(
                    ResBlock(kernel_size, conv_stack[i - 1]),
                    nn.Conv2d(conv_stack[i - 1], conv_stack[i],
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2).to(device),
                    nn.ReLU(inplace=True).to(device)
                ).to(device)
            else:
                block = ResBlock(kernel_size, conv_stack[i - 1])
            self.conv_layers.append(block)
            self.add_module("ConvBlock-" + str(i - 1), self.conv_layers[-1])

        # We have operated so far to preserve all of the spatial dimensions so
        # we can estimate the flattened dimension.
        first_fc_dim = conv_stack[-1] * IMG_SIZE[-1] * IMG_SIZE[-2]
        adjusted_fc_stack = [first_fc_dim] + fc_stack
        self.fc_layers = []
        for i in range(1, len(adjusted_fc_stack)):
            self.fc_layers.append(nn.Linear(adjusted_fc_stack[i - 1],
                                            adjusted_fc_stack[i]))
            self.add_module("FC-" + str(i - 1), self.fc_layers[-1])

    def forward(self, x):
        """
        x: batch_size x channels x Height x Width
        """
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


class IOsEncoder(nn.Module):
    def __init__(self, kernel_size, conv_stack, fc_stack):
        super(IOsEncoder, self).__init__()

        # Do one layer of convolution before stacking

        # Deduce the size of the embedding for each grid
        initial_dim = conv_stack[0] // 2  # Because we are going to get dim from I
        # and dim from O

        # we know that our grids are mostly sparse, and only positive.
        # That means that a different initialisation might be more appropriate.
        self.in_grid_enc = MapModule(nn.Sequential(
            nn.Conv2d(IMG_SIZE[0], initial_dim,
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        ), 3).to(device)
        self.out_grid_enc = MapModule(nn.Sequential(
            nn.Conv2d(IMG_SIZE[0], initial_dim,
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        ), 3).to(device)

        # Define the model that works on the stacking
        self.joint_enc = MapModule(nn.Sequential(
            GridEncoder(kernel_size, conv_stack, fc_stack)
        ), 3).to(device)

    def forward(self, input_grids, output_grids):
        """
        {input, output}_grids: batch_size x nb_ios x channels x height x width
        """
        inp_emb = self.in_grid_enc(input_grids)
        out_emb = self.out_grid_enc(output_grids)
        # {inp, out}_emb: batch_size x nb_ios x feats x height x width

        io_emb = torch.cat([inp_emb, out_emb], 2)
        # io_emb: batch_size x nb_ios x 2 * feats x height x width
        joint_emb = self.joint_enc(io_emb)
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
                 device):
        super(IOs2Seq, self).__init__()
        self.encoder = IOsEncoder(kernel_size, conv_stack, fc_stack)
        io_emb_size = fc_stack[-1]
        self.decoder = MultiIOProgramDecoder(tgt_vocabulary_size,
                                             tgt_embedding_dim,
                                             io_emb_size,
                                             decoder_lstm_hidden_size,
                                             decoder_nb_lstm_layers,
                                             device=device)

    def set_syntax_checker(self, grammar_cls):
        self.decoder.set_syntax_checker(grammar_cls)

    def forward(self, input_grids, output_grids, tgt_inp_sequences):
        io_embedding = self.encoder(input_grids, output_grids)
        dec_outs, _, _ = self.decoder(tgt_inp_sequences,
                                      io_embedding)
        return dec_outs

    def sample_model(self, input_grids, output_grids,
                     tgt_start, tgt_end, max_len,
                     nb_samples, vol=True):
        # Do the encoding of the source_sequence.
        io_embedding = self.encoder(input_grids, output_grids)

        rolls = self.decoder.sample_model(io_embedding,
                                          tgt_start, tgt_end, max_len,
                                          nb_samples, vol)
        return rolls
