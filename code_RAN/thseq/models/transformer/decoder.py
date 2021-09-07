import torch
import torch.nn as nn
import torch.nn.functional as F

import thseq
from thseq.data import Vocabulary
from thseq.modules import SinusoidalPositionEmbedding, MultiHeadAttention, \
                          PositionWiseFeedForward, maybe_norm, \
                          RandomAttention, RecRandomAttention, MultiHeadwithRandomAttention, \
                          MultiHeadwithRecRandomAttention, DeepTransitionLayer

from ..abs import _Decoder

ATTATIONS = {"std": MultiHeadAttention,
             "rand": RandomAttention,
             "recrand": RecRandomAttention,
             "attwithrand": MultiHeadwithRandomAttention,
             "attwithrecrand": MultiHeadwithRecRandomAttention
             }

class DecoderLayer(nn.Module):
    def __init__(self, args, att_type):
        super().__init__()
        hidden_size = args.hidden_size
        attention_hidden_size = args.attention_hidden_size
        ffn_hidden_size = args.ffn_hidden_size
        self.post_norm = args.decoder_post_norm
        self.ln0 = nn.LayerNorm(hidden_size)
        self.masked_self_attention = ATTATIONS[att_type](
            attention_hidden_size or hidden_size,
            args.num_heads,
            True,
            q_size=hidden_size,
            k_size=hidden_size,
            v_size=hidden_size,
            output_size=hidden_size,
            dropout=args.attention_dropout
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.encoder_decoder_attention = ATTATIONS[args.cross_att_type](
            attention_hidden_size or hidden_size,
            args.num_heads,
            False,
            q_size=hidden_size,
            k_size=hidden_size,
            v_size=hidden_size,
            output_size=hidden_size,
            dropout=args.attention_dropout
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = PositionWiseFeedForward(
            hidden_size,
            ffn_hidden_size,
            hidden_size,
            args.ffn_dropout,
            nonlinear=args.decoder_nonlinear
        )
        self.residual_dropout = args.residual_dropout

    def forward(self, x,
                encoder_output,
                self_atn_mask,
                state,
                randatt
                ):

        H = encoder_output['H']
        H_mask = encoder_output['mask']
        if state is not None:
            state['encoder'] = state.get('encoder', {})

        residual = x
        x = maybe_norm(self.ln0, x, True, self.post_norm)

        if 'decrandatt' in randatt:
            x = self.masked_self_attention(q=x, k=x, v=x,
                                           randatt=randatt['decrandatt'].unsqueeze(0),
                                           mask=self_atn_mask,
                                           state=state
                                           )
        else:
            x = self.masked_self_attention(q=x, k=x, v=x, mask=self_atn_mask, state=state)

        x = F.dropout(x, self.residual_dropout, self.training)
        x = residual + x
        x = maybe_norm(self.ln0, x, False, self.post_norm)

        residual = x
        x = maybe_norm(self.ln1, x, True, self.post_norm)
        encoder_state = state['encoder'] if state is not None else None
        if 'crossrandatt' in randatt:
            x = self.encoder_decoder_attention(
                q=x, k=H, v=H, randatt=randatt['crossrandatt'].unsqueeze(0),
                mask=H_mask,
                state=encoder_state,
            )
        else:
            x = self.encoder_decoder_attention(
                q=x, k=H, v=H, mask=H_mask,
                state=encoder_state,
            )
        x = F.dropout(x, self.residual_dropout, self.training)
        x = residual + x
        x = maybe_norm(self.ln1, x, False, self.post_norm)

        residual = x
        x = maybe_norm(self.ln2, x, True, self.post_norm)
        x = self.ffn(x)
        x = F.dropout(x, self.residual_dropout, self.training)
        x = residual + x
        x = maybe_norm(self.ln2, x, False, self.post_norm)
        return x


class Decoder(_Decoder):
    def __init__(self, args, embedding, vocabulary: Vocabulary = None):
        super().__init__(args, embedding, vocabulary)
        hidden_size = args.hidden_size
        num_layers = args.num_decoder_layers

        self.pe = SinusoidalPositionEmbedding(hidden_size)
        self.scaling = hidden_size ** 0.5

        self.dif = args.dif
        if args.dif:
            self.layer0 = DecoderLayer(args, "std")
            self.layers = nn.ModuleList([DecoderLayer(args, args.dec_att_type) for _ in range(num_layers-1)])
        else:
            self.layers = nn.ModuleList([DecoderLayer(args, args.dec_att_type) for _ in range(num_layers)])

        self.post_norm = args.decoder_post_norm
        self.ln = None
        if not self.post_norm:
            self.ln = nn.LayerNorm(hidden_size)
        self.residual_dropout = args.residual_dropout

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._self_atn_mask = torch.empty(0, 0, 0)
        self.rec_func = args.rec_func


        if args.dec_att_type in ['recrand', 'attwithrecrand']:
            maxlen = 256
            self.dec_randatt = nn.Parameter(torch.zeros(args.num_heads, maxlen, maxlen))

            nn.init.xavier_uniform_(self.dec_randatt)

            # self.dec_rnn = PositionWiseFeedForward(maxlen, 1024, maxlen)

            if self.rec_func == 0:
                self.dec_rnn = PositionWiseFeedForward(maxlen, 1024, maxlen)
            elif self.rec_func == 1:
                self.dec_rnn = nn.Sequential(nn.Linear(maxlen, maxlen), nn.Tanh())
            elif self.rec_func == 2:
                self.dec_rnn = nn.Sequential(nn.Linear(maxlen, maxlen), nn.ReLU())
            elif self.rec_func == 3:
                self.dec_rnn = nn.Sequential(nn.LayerNorm(maxlen), PositionWiseFeedForward(maxlen, 1024, maxlen))


        self.dec_att_type = args.dec_att_type

        if args.cross_att_type in ['recrand', 'attwithrecrand']:
            maxlen = 256
            self.cross_randatt = nn.Parameter(torch.zeros(args.num_heads, maxlen, maxlen))
            nn.init.xavier_uniform_(self.cross_randatt)
            self.cross_rnn = DeepTransitionLayer(maxlen, 1024, maxlen)

        self.cross_att_type = args.cross_att_type

    def forward(self, y, state):
        """
        Args:
            y: (B, T)
            state: a dictionary.
        Returns:
            (logit, state)
        """

        mask = None
        pe = None
        if self.pe is not None:
            pe = self.pe(y)
        if thseq.is_inference():
            y = self.embedding(y[:, -1:])
            pe = pe[:, -1:]
        else:
            y = self.embedding(y)
            mask = self.get_self_atn_mask(y)
        if pe is not None:
            y = y * self.scaling + pe

        y = F.dropout(y, self.residual_dropout, self.training)
        y = y.transpose(1, 0)

        randatt = {}

        if self.dec_att_type in ['recrand', 'attwithrecrand']:
            dec_prev_h = self.dec_randatt
        else:
            dec_prev_h = None

        if self.cross_att_type in ['recrand', 'attwithrecrand']:
            cross_prev_h = self.cross_randatt.clone()
        else:
            cross_prev_h = None


        j = 0

        if self.dif:
            key = f'l{j}'
            state[key] = state.get(key, {})

            y = self.layer0(
                y,
                encoder_output=state['encoder'],
                self_atn_mask=mask,
                state=state[key] if thseq.is_inference() else None,
                randatt=randatt
            )

            j += 1

        for i, layer in enumerate(self.layers):
            key = f'l{j+i}'
            state[key] = state.get(key, {})

            if self.dec_att_type in ['recrand', 'attwithrecrand']:
                res = dec_prev_h
                dec_h = self.dec_rnn(dec_prev_h)
                dec_h = dec_h + res
                randatt['decrandatt'] = dec_h
                dec_prev_h = dec_h

            if self.cross_att_type in ['recrand', 'attwithrecrand']:
                cross_h, _ = self.cross_rnn(cross_prev_h)
                cross_prev_h = cross_h
                randatt['crossrandatt'] = cross_h

            y = layer(
                y,
                encoder_output=state['encoder'],
                self_atn_mask=mask,
                state=state[key] if thseq.is_inference() else None,
                randatt=randatt
            )

        if thseq.is_inference():
            del state['encoder']['H']
            state['encoder']['H'] = None
        if self.ln is not None:
            y = self.ln(y)

        y = y.transpose(1, 0)
        y = self.logit(y)
        return y, state

    def get_self_atn_mask(self, y):
        _, T, _ = y.size()
        if self._self_atn_mask.size(1) < T:
            self._self_atn_mask = y.new_full((T, T), float('-inf')).triu(1).unsqueeze(0)
        self._self_atn_mask = self._self_atn_mask.to(y.device)
        return self._self_atn_mask[:, :T, :T]
