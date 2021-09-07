import torch
import torch.nn as nn
import torch.nn.functional as F

from thseq.data import Vocabulary
from thseq.models.abs import _Encoder
from thseq.modules import SinusoidalPositionEmbedding, \
                          MultiHeadAttention, \
                          PositionWiseFeedForward, \
                          maybe_norm, \
                          RandomAttention, RecRandomAttention, MultiHeadwithRandomAttention, \
                          MultiHeadwithRecRandomAttention, FactorizedRandomAttention

ATTATIONS = {"std": MultiHeadAttention,
             "rand": RandomAttention,
             "recrand": RecRandomAttention,
             "attwithrand": MultiHeadwithRandomAttention,
             "attwithrecrand": MultiHeadwithRecRandomAttention,
             "facrand": FactorizedRandomAttention
             }

class EncoderLayer(nn.Module):
    def __init__(self, args, att_type):
        super().__init__()
        hidden_size = args.hidden_size
        attention_hidden_size = args.attention_hidden_size
        ffn_hidden_size = args.ffn_hidden_size
        self.post_norm = args.encoder_post_norm

        self.ln0 = nn.LayerNorm(hidden_size)

        self.self_attention = ATTATIONS[att_type](
            attention_hidden_size or hidden_size,
            args.num_heads,
            False,
            q_size=hidden_size,
            k_size=hidden_size,
            v_size=hidden_size,
            output_size=hidden_size,
            dropout=args.attention_dropout
        )

        self.ln1 = nn.LayerNorm(hidden_size)

        self.ffn = PositionWiseFeedForward(
            hidden_size,
            ffn_hidden_size,
            hidden_size,
            args.ffn_dropout,
            nonlinear=args.encoder_nonlinear
        )

        self.residual_dropout = args.residual_dropout

        self.enc_att_type = att_type

    def forward(self, x, mask, randatt=None):
        residual = x
        x = maybe_norm(self.ln0, x, True, self.post_norm)

        if self.enc_att_type in ['recrand','attwithrecrand']:
            x = self.self_attention(q=x, k=x, v=x, randatt=randatt, mask=mask)
        else:
            x = self.self_attention(q=x, k=x, v=x, mask=mask)

        x = F.dropout(x, self.residual_dropout, self.training)
        x = residual + x
        x = maybe_norm(self.ln0, x, False, self.post_norm)

        residual = x
        x = maybe_norm(self.ln1, x, True, self.post_norm)
        x = self.ffn(x)
        x = F.dropout(x, self.residual_dropout, self.training)
        x = residual + x
        x = maybe_norm(self.ln1, x, False, self.post_norm)
        return x

class Encoder(_Encoder):
    def __init__(self, args, embedding, vocabulary: Vocabulary = None):
        super().__init__(args, embedding, vocabulary)
        hidden_size = args.hidden_size
        num_layers = args.num_encoder_layers

        self.pe = SinusoidalPositionEmbedding(hidden_size)
        self.scaling = hidden_size ** 0.5

        self.dif = args.dif

        if args.dif:
            self.layer0 = EncoderLayer(args, "std")
            self.layers = nn.ModuleList([EncoderLayer(args, args.enc_att_type) for _ in range(num_layers-1)])
        else:
            self.layers = nn.ModuleList([EncoderLayer(args, args.enc_att_type) for _ in range(num_layers)])

        self.post_norm = args.encoder_post_norm
        self.ln = None
        if not self.post_norm:
            self.ln = nn.LayerNorm(hidden_size)
        self.residual_dropout = args.residual_dropout

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        maxlen = 256
        self.rec_func = args.rec_func

        if args.enc_att_type in ['recrand', 'attwithrecrand']:

            self.randatt = nn.Parameter(torch.zeros(args.num_heads, maxlen, maxlen))
            nn.init.xavier_uniform_(self.randatt)

            if self.rec_func == 0:
                self.rnn = PositionWiseFeedForward(maxlen, 1024, maxlen)
            elif self.rec_func == 1:
                self.rnn = nn.Sequential(nn.Linear(maxlen, maxlen), nn.Tanh())
            elif self.rec_func == 2:
                self.rnn = nn.Sequential(nn.Linear(maxlen, maxlen), nn.ReLU())
            elif self.rec_func == 3:
                self.rnn = nn.Sequential(nn.LayerNorm(maxlen), PositionWiseFeedForward(maxlen, 1024, maxlen))

        self.enc_att_type = args.enc_att_type
        self.num_heads = args.num_heads
        self.maxlen = maxlen

    def forward(self, x):
        """
        Args:
            x: (B, T)

        Returns:
            state: a dictionary.
        """
        B, T = x.shape
        mask = x.eq(self.embedding.padding_idx).unsqueeze(1)
        x = self.embedding(x) * self.scaling + self.pe(x)
        x = F.dropout(x, self.residual_dropout, self.training)

        if self.enc_att_type in ['recrand', 'attwithrecrand']:
            prev_h = self.randatt
        else:
            prev_h = None

        x = x.transpose(1, 0)

        if self.dif:
            x = self.layer0(x, mask)

        for idx, layer in enumerate(self.layers):
            if self.enc_att_type in ['recrand', 'attwithrecrand']:
                res = prev_h
                h = self.rnn(prev_h)
                h = res + h
                x = layer(x, mask, randatt=h.unsqueeze(0))
                prev_h = h
            else:
                x = layer(x, mask)

        if self.ln is not None:
            x = self.ln(x)
        state = {
            'encoder': {
                'H': x,
                'mask': mask
            }
        }
        return state
