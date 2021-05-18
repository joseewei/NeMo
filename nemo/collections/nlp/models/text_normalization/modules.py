import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
# DEVICE=torch.device('cuda:0') # or set to 'cpu'

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""

        encoder_outputs, encoder_hidden = self.encode(src, src_lengths)
        decoder_output, decoder_hidden = self.decode(encoder_outputs, encoder_hidden, src_lengths, trg, trg_lengths)
        pre_output = self.generator.proj(decoder_output)
        return pre_output

    def encode(self, src, src_lengths):
        return self.encoder(self.src_embed(src), src_lengths)

    def decode(self, encoder_outputs, encoder_hidden, src_lengths, trg, trg_lengths):
        return self.decoder(encoder_outputs, encoder_hidden, src_lengths, self.trg_embed(trg), trg_lengths)


class EncoderRNN(nn.Module):

    """Encodes a sequence of word embeddings
    
    Args:
        input_size: size of embedded input
        hidden_size: hidden size of encoded input after send through encoder
        dropout: dropout rate
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden: Optional[torch.Tensor] = None):
        '''
        Applies a bidirectional GRU to sequence of embeddings input_seqs.
        The input mini-batch input_seqs needs to be sorted by length.
        input_seqs should have dimensions [time, batch, dim].

        Args:
            input_seqs: shape (B, T, E) sorted decreasingly by lengths(for packing)
            input_lengths: list of sequence length (B, 1)
            hidden: initial state of GRU, can be left out
        
        Returns:
            GRU outputs in shape (batch, seq_len, 2 * hidden_size)
            last hidden state of RNN(i.e. last output for GRU) (num_layers, batch, 2 * hidden_size)
        '''
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=input_seqs, lengths=input_lengths, batch_first=True, enforce_sorted=True
        )
        # outputs (seq_len, batch, 2* hidden_size)
        # hidden (num_layers * 2, batch, hidden_size)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs.transpose(0, 1).contiguous()  # (B, T, 2*H)

        fwd_h_final = hidden[0 : hidden.size(0) : 2]
        bwd_h_final = hidden[1 : hidden.size(0) : 2]
        hidden = torch.cat([fwd_h_final, bwd_h_final], dim=2)  # [num_layers, batch, 2*hidden_size]
        return outputs, hidden


class DynamicEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden: Optional[torch.Tensor] = None):
        '''
        Applies a bidirectional GRU to sequence of embeddings input_seqs.
        The input mini-batch input_seqs do NOT need to be sorted by length.
        input_seqs should have dimensions [batch, time, dime].

        Args:
            input_seqs: shape (seq_len, batch, input_size) sorted decreasingly by lengths(for packing)
            input_lengths: list of sequence length
            hidden: initial state of GRU, can be left out
        
        Returns:
            GRU outputs in shape (B, T, 2 * hidden_size)
            last hidden state of RNN(i.e. last output for GRU) (num_layers * 2, batch, hidden_size)
        '''
        sort_idx = np.argsort(-input_lengths)
        unsort_idx = torch.LongTensor(np.argsort(sort_idx)).cuda()
        input_lengths = input_lengths[sort_idx]
        sort_idx = torch.LongTensor(sort_idx).cuda()
        input_seqs = input_seqs[:, sort_idx, :]
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            input_seqs, input_lengths, batch_first=True, enforce_sorted=False
        )

        # outputs (seq_len, batch, 2* hidden_size)
        # hidden (num_layers * 2, batch, hidden_size)

        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        fwd_h_final = hidden[0 : hidden.size(0) : 2]
        bwd_h_final = hidden[1 : hidden.size(0) : 2]
        hidden = torch.cat([fwd_h_final, bwd_h_final], dim=2)  # [num_layers, batch, 2*hidden_size]

        outputs = outputs[:, unsort_idx, :]
        hidden = hidden[:, unsort_idx, :]

        # outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        # hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden


class Attn(nn.Module):
    """
    concat attention
    """

    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden_state, encoder_outputs, src_len=None):
        """
        Args:
            hidden_state: previous decode hidden state, in shape (B,H)
            encoder_outputs: encoder outputs from encoder, in shape (B,T, H)
            src_len: used for masking. NoneType or tensor in shape (B) indicating sequence length
        returns:
            attention energies in shape (B,T)
        """
        max_len = encoder_outputs.size(1)
        H = hidden_state.unsqueeze(1).repeat(1, max_len, 1)  # [B,T,H]
        attn_energies = self.score(H, encoder_outputs)  # compute attention score

        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = cuda_(torch.ByteTensor(mask).unsqueeze(1))  # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)

        return F.softmax(attn_energies, dim=-1)

    def score(self, hidden_state, encoder_outputs):
        """
        Args:
            hidden_state: (B, T,  H)
            encoder_outputs: (B, T, H)
        """
        energy = F.tanh(self.attn(torch.cat([hidden_state, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, num_layers=1, dropout=0.1, bridge=True):
        super(DecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.dropout = dropout
        # Define layers
        self.dropout = nn.Dropout(dropout)
        # to initialize from the final encoder state
        self.bridge = nn.Linear(hidden_size, hidden_size, bias=True) if bridge else None

        self.attn = Attn(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        # self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)

        # pre output layer
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, encoder_outputs, last_hidden, src_lengths, trg, trg_lengths, max_len=None):
        """
        Args:
            encoder_outputs:
                encoder outputs in shape (B, T, 2*H)
            last_hidden:
                last hidden stat of the encoder, in shape (layers, B, 2 * H)
            src_lengths:
                encoder lenghts (B, 1)
            trg:
                embedded input for current time step, in shape (B, T, E)
            trg_lengths: (B, 1)
            max_len:  int
        return:
            decoder output
        """

        bs = trg.size(0)
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = int(trg_lengths.max(dim=-1)[0].item())

        hidden = self.init_hidden(last_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []

        for i in range(max_len):
            prev_embed = trg[:, i].unsqueeze(1)  # [B, 1, E]
            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attn(hidden[-1], encoder_outputs)  # (B, T)
            context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  # (B,1,2 *H)
            # Combine embedded input word and attended context, run through RNN
            rnn_input = torch.cat((prev_embed, context), 2)  # (B, 1, embed_size + 2 * H)
            # rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
            # rnn_input = F.relu(rnn_input)
            # output = (B, 1, 2*H)
            # hidden (layers, B, 2*H)
            output, hidden = self.gru(rnn_input, last_hidden)

            decoder_states.append(output)

        decoder_states = torch.cat(decoder_states, dim=1)  # (B, T, 2*H)
        return decoder_states, hidden

    def init_hidden(self, encoder_hidden):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""
        if encoder_hidden is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_hidden))


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)


if __name__ == "__main__":
    encoder_input_size = 20
    decoder_input_size = 20
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    hidden_size = 5
    bs = 8
    max_seq_length = 10
    dropout = 0.1
    num_layers = 2

    model = EncoderDecoder(
        EncoderRNN(input_size=encoder_input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout),
        DecoderRNN(embed_size=decoder_input_size, hidden_size=2 * hidden_size, num_layers=num_layers, dropout=dropout),
        nn.Embedding(src_vocab_size, encoder_input_size),
        nn.Embedding(tgt_vocab_size, decoder_input_size),
        Generator(hidden_size=2 * hidden_size, vocab_size=tgt_vocab_size),
    )

    input_seqs = torch.randint(high=src_vocab_size, size=(bs, max_seq_length))
    input_lengths = torch.ones(bs) * max_seq_length

    output_seqs = torch.randint(high=tgt_vocab_size, size=(bs, max_seq_length))
    output_lengths = torch.ones(bs) * max_seq_length

    output = model(src=input_seqs, trg=output_seqs, src_lengths=input_lengths, trg_lengths=output_lengths)
    import ipdb

    ipdb.set_trace()
