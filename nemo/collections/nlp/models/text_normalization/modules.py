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

    def __init__(self, encoder, decoder, src_embed, trg_embed, teacher_forcing):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed

        self.trg_embed = trg_embed
        self.teacher_forcing = teacher_forcing

        self.src_embed.weight.data.normal_(0, 0.1)

        self.trg_embed.weight.data.normal_(0, 0.1)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_lengths: torch.Tensor,
        decoder_init_hidden: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
    ):
        """
        Args: 
            src: input embedding. padded part should be masked out (B, T)
            tgt: output embedding. padded part should be masked out (B, T)
            src_lengths: input length, (B)
            decoder_init_hidden: initial hidden state for decoder, (1, B, Hd)
            max_len: max output length, only specified for inference
        """
        encoder_outputs, encoder_hidden = self.encoder(self.src_embed(src), src_lengths)
        if decoder_init_hidden is None:
            decoder_init_hidden = encoder_hidden

        #### prepare decoding
        bs = encoder_outputs.size(0)
        src_max_length = encoder_outputs.size(1)
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg.size(1)
        if (not self.training) or (np.random.rand() > self.teacher_forcing):
            use_teacher_forcing = False
        else:
            use_teacher_forcing = True

        hidden = self.decoder.init_hidden(decoder_init_hidden)

        # here we store all intermediate hidden states and pre-output vectors

        decoder_states = torch.zeros(bs, max_len, self.decoder.num_classes, device=encoder_outputs.device)
        src_mask = torch.arange(src_max_length, device=encoder_outputs.device).expand(
            bs, src_max_length
        ) < src_lengths.unsqueeze(1)

        trg = self.trg_embed(trg)
        decoder_input = trg[:, 0].unsqueeze(1)
        # unroll as long as max_len
        for i in range(max_len):
            output, hidden = self.decoder(
                encoder_outputs=encoder_outputs,
                encoder_mask=src_mask,
                decoder_input=decoder_input,
                prev_decoder_hidden=hidden,
            )
            decoder_states[:, i : i + 1, :] = output

            if use_teacher_forcing:
                decoder_input = trg[:, i].unsqueeze(1)
            else:
                decoder_input = torch.argmax(output, dim=2).detach()
                decoder_input = self.trg_embed(decoder_input)

        return decoder_states


class EncoderRNN(nn.Module):

    """Encodes a sequence of word embeddings
    
    Args:
        input_size: size of embedded input
        hidden_size: hidden size of encoded input after send through encoder
        num_layers: number of layers
        dropout: dropout rate
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden: Optional[torch.Tensor] = None):
        '''
        Applies a bidirectional GRU to sequence of embeddings input_seqs.

        Args:
            input_seqs: embedded input (B, T, D)
            input_lengths: list of sequence length (B)
            hidden: optional, initial hidden state of GRU
        
        Returns:
            GRU output (B, T, 2*H)
            last hidden state of RNN (num_layers, B, 2*H)
        '''
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=input_seqs, lengths=input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # outputs (T, B, 2*H)
        # hidden (num_layers * 2, B, H)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # dropout
        outputs = outputs.transpose(0, 1).contiguous()  # (B, T, 2*H)

        fwd_h_final = hidden[0 : hidden.size(0) : 2]
        bwd_h_final = hidden[1 : hidden.size(0) : 2]
        hidden = torch.cat([fwd_h_final, bwd_h_final], dim=2)  # (num_layers, B, 2*)
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
            input_seqs, input_lengths, batch_first=True, enforce_sorted=True
        )

        # outputs (T, B, 2*H)
        # hidden (num_layers * 2, B, H)

        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        fwd_h_final = hidden[0 : hidden.size(0) : 2]
        bwd_h_final = hidden[1 : hidden.size(0) : 2]
        hidden = torch.cat([fwd_h_final, bwd_h_final], dim=2)  # (num_layers, B, 2*)

        outputs = outputs[:, unsort_idx, :]
        hidden = hidden[:, unsort_idx, :]

        # outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        # hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden


class Attention(nn.Module):
    """
    concat attention

    Args:
        attention_hidden_size: attention hidden size
        encoder_hidden_size: encoder hidden size
        decoder_hidden_size: decoder hidden size
    """

    def __init__(self, attention_hidden_size: int, encoder_hidden_size: int, decoder_hidden_size: int):
        super(Attention, self).__init__()

        self.attn = nn.Linear(2 * encoder_hidden_size + decoder_hidden_size, attention_hidden_size)
        self.v = nn.Parameter(torch.rand(attention_hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(
        self, hidden_state: torch.Tensor, encoder_outputs: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            hidden_state: query: decode hidden state, in shape (B,H)
            encoder_outputs: keys, encoder outputs from encoder, in shape (B,T, H)
            src_len: used for masking. NoneType or tensor in shape (B) indicating sequence length
        returns:
            attention energies in shape (B,T)
        """
        max_len = encoder_outputs.size(1)
        H = hidden_state.unsqueeze(1).repeat(1, max_len, 1)  # (B,T,H)
        attn_energies = self.score(H, encoder_outputs)  # compute attention score (B, T)

        if src_mask is not None:
            attn_energies = attn_energies.masked_fill(src_mask == 0, -float('inf'))

        return F.softmax(attn_energies, dim=-1)

    def score(self, hidden_state: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        Computes unnormalized attention weights 

        Args:
            hidden_state: usually decoder hidden state - query (B, T, H)
            encoder_outputs: encoder outputs - key (B, T, H)
        """
        energy = F.tanh(self.attn(torch.cat([hidden_state, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class DecoderAttentionRNN(nn.Module):
    """

    Args:
        teacher forcing ratio:  1 means full teacher forcing , 0 means no teacher forcing e.g. used for inference

    """

    def __init__(
        self,
        attention: nn.Module,
        hidden_size: int,
        embed_size: int,
        num_classes: int,
        num_layers=1,
        dropout=0.1,
        teacher_forcing=1.0,
        bridge=False,
    ):
        super(DecoderAttentionRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        # Define layers
        self.dropout = nn.Dropout(dropout)
        self.teacher_forcing = teacher_forcing
        # to initialize from the final encoder state
        self.bridge = nn.Linear(hidden_size, hidden_size, bias=True) if bridge else None

        self.attn = attention
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        # first hidden size should be 2 * encoder hidden size, in this case its the same
        self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.proj = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, encoder_outputs, encoder_mask, decoder_input, prev_decoder_hidden):
        """
        only single step
        Args:
            encoder_outputs: encoder output (B, T, He*2)        
            encoder_mask: (B, T)
            decoder_input: (B, 1, D)
            prev_decoder_hidden: (num_layers, B, H)
        """
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(
            hidden_state=prev_decoder_hidden[-1], encoder_outputs=encoder_outputs, src_mask=encoder_mask
        )  # (B, T)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  # (B,1,2*He)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((decoder_input, context), 2)  # (B, 1, H + He*2)
        rnn_input = self.attn_combine(rnn_input)  # (B, 1, H)
        # rnn_input = F.tanh(rnn_input) #makes it worse
        # output = (B, 1, H)
        # hidden (layers, B, H)
        output, hidden = self.gru(rnn_input, prev_decoder_hidden)
        output = self.dropout(output)
        output = self.proj(output)
        return output, hidden

    def init_hidden(self, hidden):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""
        if hidden is None:
            return None  # start with zeros
        if self.bridge:
            return torch.tanh(self.bridge(hidden))
        else:
            return hidden


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden


