# Copyright 2020 University of Toronto, all rights reserved

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


class Encoder(EncoderBase):

    def init_submodules(self):
        # initialize parameterized submodules here: rnn, embedding
        # using: self.source_vocab_size, self.word_embedding_size, self.pad_id,
        # self.dropout, self.cell_type, self.hidden_state_size,
        # self.num_hidden_layers
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # relevant pytorch modules:
        # torch.nn.{LSTM, GRU, RNN, Embedding}
        
        # Adding padding_idx enables "It must not learn an embedding for padded tokens."
        self.embedding = torch.nn.Embedding(self.source_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)
        if self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size, num_layers=self.num_hidden_layers, dropout=self.dropout, bidirectional=True)
        elif self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size, num_layers=self.num_hidden_layers, dropout=self.dropout,bidirectional=True)
        elif self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size, num_layers=self.num_hidden_layers, dropout=self.dropout,bidirectional=True)

    def get_all_rnn_inputs(self, F):
        # compute input vectors for each source transcription.
        # F is shape (S, N)
        # x (output) is shape (S, N, I)
        # assert False, "Fill me"

        embedded = self.embedding(F)
        # Check Whenever ``s`` exceeds the original length of ``F[s, n]`` (i.e.
        # when ``F[s, n] == self.pad_id``), ``x[s, n, :] == 0.``
        pad_id_index = (F == self.pad_id).nonzero()
        for idx in pad_id_index:
            # print(idx)
            if not torch.all(torch.eq(embedded[idx[0], idx[1], :],torch.zeros(self.word_embedding_size))):
                print("WRONG: PADDING ROW IS EQUAL TO ZERO!!!!!!!")
                exit()

        print("Embeddeing layer size is " + str(embedded.size()))

        # Add a dropout layer on the input layer
        # dropout = torch.nn.dropout(self.dropout)
        # embedded = dropout(self.embedding(F))
        
        return embedded

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # compute all final hidden states for provided input sequence.
        # make sure you handle padding properly!
        # x is of shape (S, N, I)
        # F_lens is of shape (N,)
        # h_pad is a float
        # h (output) is of shape (S, N, 2 * H)
        # relevant pytorch modules:
        # torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        
        # packed the input 
        # Sorted or not ????
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens)

        # for gru h0 ????????????????????
        # split lstm and gru/rnn ?????
        # ADD hidden layer ???????????
        if self.cell_type == 'lstm':
            packed_output, _, _ = self.rnn(packed_input)
        else:
            packed_output, _ = self.rnn(packed_output)

        #Unpacked the output with the h_pad
        h = torch.nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=h_pad)
        
        return h


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # initialize parameterized submodules: embedding, cell, ff
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # relevant pytorch modules:
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # torch.nn.{Embedding,Linear,LSTMCell,RNNCell,GRUCell}
        self.embedding = torch.nn.Embedding(self.target_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size)
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size)
        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)


    def get_first_hidden_state(self, h, F_lens):
        # build decoder's first hidden state. Ensure it is derived from encoder
        # hidden state that has processed the entire sequence in each
        # direction:
        # - Populate indices 0 to self.hidden_state_size // 2 - 1 (inclusive)
        #   with the hidden states of the encoder's forward direction at the
        #   highest index in time *before padding*
        # - Populate indices self.hidden_state_size // 2 to
        #   self.hidden_state_size - 1 (inclusive) with the hidden states of
        #   the encoder's backward direction at time t=0
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # htilde_tm1 (output) is of shape (N, 2 * H)
        # relevant pytorch modules: torch.cat

        # Get the last hidden_layer's ouput of the encoder for forward direction
        forward = h[-1,:, :self.hidden_state_size//2]

        # Get the first hidden_layer's output of the encoder for backward direction
        backward = h[0,:, self.hidden_state_size//2:]

        htilde_tm1 = torch.cat((forward, backward), 1)

        return htilde_tm1


    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # determine the input to the rnn for *just* the current time step.
        # No attention.
        # E_tm1 is of shape (N,)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # xtilde_t (output) is of shape (N, Itilde)
        
        #E_tm1 [1, N]
        E_tm1 = E_tm1.unsqueeze(0)

        embedded = self.embedding(E_tm1)
        xtilde_t = embedded

        return xtilde_t
        


    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # update the previous hidden state to the current hidden state.
        # xtilde_t is of shape (N, Itilde)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # htilde_t (output) is of same shape as htilde_tm1
        if self.cell_type == "lstm":
            htilde_tm1 = self.cell(xtilde_t, htilde_tm1[0], htilde_tm1[1])
        else:
            htilde_t = self.cell(xtilde_t, htilde_tm1)
            # Check htilde_t (output) is of same shape as htilde_tm1
            if not htilde_tm1.size().eq(htilde_t.size()):
                print("WRROR NOT THE SAME DIMENSION BETWEEN htilde_t and htilde_tm1")
                exit()

        return htilde_t


    def get_current_logits(self, htilde_t):
        # determine un-normalized log-probability distribution over output
        # tokens for current time step.
        # htilde_t is of shape (N, 2 * H), even for LSTM (cell state discarded)
        # logits_t (output) is of shape (N, V)
        logits_t = self.ff(htilde_t)
        print(logits_t.size())
    
        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # same as before, but with a slight modification for attention
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # cell_type will be one of: ['lstm', 'gru', 'rnn']

    def get_first_hidden_state(self, h, F_lens):
        # same as before, but initialize to zeros
        # relevant pytorch modules: torch.zeros_like
        # ensure result is on same device as h!
        assert False, "Fill me"

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # update to account for attention. Use attend() for c_t
        assert False, "Fill me"

    def attend(self, htilde_t, h, F_lens):
        # compute context vector c_t. Use get_attention_weights() to calculate
        # alpha_t.
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # c_t (output) is of shape (N, 2 * H)
        assert False, "Fill me"

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of shape (S, N)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, N)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Determine energy scores via cosine similarity
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # e_t (output) is of shape (S, N)
        assert False, "Fill me"


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # initialize the parameterized submodules: encoder, decoder
        # encoder_class and decoder_class inherit from EncoderBase and
        # DecoderBase, respectively.
        # using: self.source_vocab_size, self.source_pad_id,
        # self.word_embedding_size, self.encoder_num_hidden_layers,
        # self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        # self.target_vocab_size, self.target_eos
        # Recall that self.target_eos doubles as the decoder pad id since we
        # never need an embedding for it
        assert False, "Fill me"

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # get logits over entire E. logits predict the *next* word in the
        # sequence.
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # E is of shape (T, N)
        # logits (output) is of shape (T - 1, N, Vo)
        # relevant pytorch modules: torch.{zero_like,stack}
        # hint: recall an LSTM's cell state is always initialized to zero.
        # Note logits sequence dimension is one shorter than E (why?)
        assert False, "Fill me"

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        # htilde_t is of shape (N, K, 2 * H) or a tuple of two of those (LSTM)
        # logpb_tm1 is of shape (N, K)
        # b_tm1_1 is of shape (t, N, K)
        # b_t_0 (first output) is of shape (N, K, 2 * H) or a tuple of two of
        #                                                         those (LSTM)
        # b_t_1 (second output) is of shape (t + 1, N, K)
        # logpb_t (third output) is of shape (N, K)
        # relevant pytorch modules:
        # torch.{flatten,topk,unsqueeze,expand_as,gather,cat}
        # hint: if you flatten a two-dimensional array of shape z of (A, B),
        # then the element z[a, b] maps to z'[a*B + b]
        assert False, "Fill me"
