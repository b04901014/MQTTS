import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AlibiPostionEmbedding:
    def __init__(self, nheads, maxpos):
        context_position = torch.arange(maxpos)[:, None]
        memory_position = torch.arange(maxpos)[None, :]
        relative_position = memory_position - context_position
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(nheads, -1,-1)
        self.slopes = torch.Tensor(self.get_slopes(nheads)) * -1
        self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * relative_position
        self.alibi = self.alibi.view(nheads, maxpos, maxpos)

    def get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
        else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround.
            return get_slopes_power_of_2(closest_power_of_2) + self.get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    def __call__(self, x):
        #N, T, C
        return self.alibi[:, :x.size(1), :x.size(1)].to(x.device)


class ABSPositionEmbedding(nn.Module):
    def __init__(self, maxlength, ndim, scaled=False):
        super().__init__()
        p = torch.zeros((maxlength, ndim))
        pi = torch.arange(start=0, end=maxlength).float().unsqueeze(1)
        pi = pi * torch.exp(torch.arange(start=0, end=ndim, step=2).float() * -(np.log(10000.0) / ndim))
        p[:, 0::2] = torch.sin(pi)
        p[:, 1::2] = torch.cos(pi)
        self.register_buffer('p', p)
        self.scaled = scaled
        self.scalar = nn.Parameter(torch.FloatTensor([1.0])) if scaled else 1.0

    def forward(self, x):
        B, L, C = x.size()
        p = self.p[: L].unsqueeze(0).expand(B, -1, -1)
        x = x + self.scalar * p
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, softmax_temp=1.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.softmax_temp = softmax_temp

    def reshape(self, x):
        x = x.view(x.size(0), x.size(1), self.nhead, self.head_dim).transpose(1, 2).contiguous()
        x = x.view(-1, x.size(2), self.head_dim)
        return x

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, attn_bias=None, past_kv=None):
        batch_size = q.size(0)
        q = self.q_proj(q) * self.head_dim ** -0.5
        if past_kv is not None:
            k, v = torch.cat([past_kv, k], 1), torch.cat([past_kv, v], 1)
        k, v = self.k_proj(k), self.v_proj(v)
        #Reshape for heads (B*nH, T, C)
        q, k, v = self.reshape(q), self.reshape(k), self.reshape(v)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (batch_size * self.nhead, q.size(1), k.size(1))
        if attn_bias is not None:
            assert attn_bias.size() == (self.nhead, q.size(1), k.size(1)), f"Should be {(self.nhead, q.size(1), k.size(1))}. Got {attn_bias.size()}"
            attn_weights = attn_weights + attn_bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(batch_size * self.nhead, q.size(1), k.size(1))
        if attn_mask is not None:
            assert attn_mask.size() == (q.size(1), k.size(1)), f"Should be {(q.size(1), k.size(1))}. Got {attn_mask.size()}"
            assert attn_mask.dtype == torch.bool
            attn_mask = attn_mask.unsqueeze(0).expand(batch_size * self.nhead, -1, -1)
        if key_padding_mask is not None:
            assert key_padding_mask.size() == (batch_size, k.size(1)), f"Should be {(batch_size, k.size(1))}. Got {key_padding_mask.size()}"
            assert key_padding_mask.dtype == torch.bool
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.nhead, -1, -1)
            key_padding_mask = key_padding_mask.reshape(batch_size * self.nhead, 1, k.size(1))
            if attn_mask is None:
                attn_mask = key_padding_mask.expand(-1, q.size(1), -1)
            else:
                attn_mask = attn_mask.logical_or(key_padding_mask)
        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights * self.softmax_temp, dim=-1, dtype=attn_weights.dtype)
        attn_weights_reshaped = attn_weights.view(batch_size, self.nhead, q.size(1), k.size(1))
#        attn_weights = attn_weights_reshaped.view(batch_size * self.nhead, q.size(1), k.size(1))
        attn_probs = self.dropout(attn_weights)
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (batch_size * self.nhead, q.size(1), self.head_dim)
        attn_output = attn_output.view(batch_size, self.nhead, q.size(1), self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q.size(1), self.d_model)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped

class CrossAttnOnlyLayer(nn.Module):
    def __init__(self, hp, dropout=0.1):
        super().__init__()
        self.dropout_p = dropout
        #Only one head for alignment!
        self.multihead_attn = MultiheadAttention(hp.hidden_size, 1, dropout=0.1, softmax_temp=hp.aligner_softmax_temp)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hp.hidden_size, hp.ffd_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hp.ffd_size, hp.hidden_size)

        self.norm1 = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.norm2 = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, tgt, memory, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, attn = self.multihead_attn(tgt, memory, memory,
                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt, attn

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hp, with_cross_attention, dropout=0.1):
        super().__init__()
        self.hp = hp
        self.d_model = hp.hidden_size
        self.dropout_p = dropout
        self.self_attn = MultiheadAttention(self.d_model, hp.nheads, dropout=0.1)
        self.with_cross_attention = with_cross_attention
        if with_cross_attention:
            self.multihead_attn = MultiheadAttention(self.d_model, hp.nheads, dropout=0.1)
            self.norm2 = nn.LayerNorm(self.d_model, eps=hp.layer_norm_eps)
            self.dropout2 = nn.Dropout(dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(self.d_model, hp.ffd_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hp.ffd_size, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model, eps=hp.layer_norm_eps)
        self.norm3 = nn.LayerNorm(self.d_model, eps=hp.layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt, memory=None, tgt_mask=None, attn_bias=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, past_kv=None):
        tgt2, self_attn = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, attn_bias=attn_bias,
                                         key_padding_mask=tgt_key_padding_mask, past_kv=past_kv)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        attn = None
        if self.with_cross_attention:
            assert memory is not None
            tgt2, attn = self.multihead_attn(tgt, memory, memory,
                                             key_padding_mask=memory_key_padding_mask, past_kv=past_kv)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn, self_attn

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layers):
        super().__init__()
        self.layers = decoder_layers
        self.num_layers = len(decoder_layers)

    def forward(self, tgt, memory, tgt_mask=None, attn_bias=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, past_kvs=None):
        output = tgt
        attns = []
        self_attns = []
        outputs = []
        if past_kvs is None:
            past_kvs = [None for _ in range(len(self.layers))]
        for i, mod in enumerate(self.layers):
            output, attn, self_attn = mod(output, memory, tgt_mask=tgt_mask, attn_bias=attn_bias,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask,
                                          past_kv=past_kvs[i])
            if attn is not None:
                attns.append(attn.detach())
            if self_attn is not None:
                self_attns.append(self_attn.detach())
            outputs.append(output)
        return output, attns, self_attns, outputs

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hp, dropout=0.1):
        super().__init__()
        self.hp = hp
        self.d_model = hp.hidden_size
        self.dropout_p = dropout
        self.self_attn = MultiheadAttention(self.d_model, hp.nheads, dropout=0.1)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(self.d_model, hp.ffd_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hp.ffd_size, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model, eps=hp.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.d_model, eps=hp.layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, attn_bias=None, src_key_padding_mask=None):
        res, self_attn = self.self_attn(src, src, src, attn_mask=src_mask, attn_bias=attn_bias,
                                        key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(res)
        src = self.norm1(src)
        res = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(res)
        src = self.norm2(src)
        return src, self_attn

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layers):
        super().__init__()
        self.layers = encoder_layers
        self.num_layers = len(encoder_layers)

    def forward(self, src, mask=None, attn_bias=None, src_key_padding_mask=None):
        output = src
        attns = []
        for i, mod in enumerate(self.layers):
            output, attn = mod(output, src_mask=mask, attn_bias=attn_bias, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn.detach())
        return output, attns
