import torch.nn as nn
import torch
from .transformers import TransformerDecoder, TransformerDecoderLayer
from argparse import Namespace

class ARCodeTransformer(nn.Module):
    def __init__(self, hp, n_decoder_codes):
        super().__init__()
        self.hp = hp
        ar_hp = Namespace(hidden_size=hp.ar_hidden_size, nheads=hp.ar_nheads, layer_norm_eps=hp.layer_norm_eps,
                          ffd_size=hp.ar_ffd_size)
        self.model = TransformerDecoder(
            nn.ModuleList(
                [TransformerDecoderLayer(ar_hp, with_cross_attention=False) for i in range(hp.ar_layer)]
            )
        )
        self.embedding = nn.ModuleList(
            [
                nn.Embedding(n_decoder_codes, hp.ar_hidden_size) for _ in range(self.hp.n_cluster_groups - 1)
            ]
        )
        self.linear = nn.Linear(hp.hidden_size, hp.ar_hidden_size)
        self.layer_norm = nn.LayerNorm(hp.ar_hidden_size, eps=hp.layer_norm_eps)
        self.decoders = nn.ModuleList([
            nn.Linear(hp.ar_hidden_size, n_decoder_codes)
            for i in range(hp.n_cluster_groups)
        ])
        tgt_mask = (torch.tril(torch.ones(hp.n_cluster_groups, hp.n_cluster_groups), diagonal=0) == 0)
        self.register_buffer('tgt_mask', tgt_mask)

    def forward(self, cond, gt):
        #cond: N, T, C
        #gt: N, T, 4
        #return: N, T, 4, n_codes
        N, T, _ = cond.size()
        cond, gt = cond.reshape(N * T, -1), gt.reshape(N * T, -1)
        cond = self.linear(cond)
        gt = gt[:, : -1] #NT, 3
        gt_in = []
        for i in range(self.hp.n_cluster_groups - 1):
            gt_in.append(self.embedding[i](gt[:, i])) #3 [NT, C]
        inp = torch.stack([cond] + gt_in, 1) #NT, 4, C
        inp = self.layer_norm(inp)
        out, _, _, _ = self.model(inp, memory=None, tgt_mask=self.tgt_mask)
        ret = []
        for i in range(self.hp.n_cluster_groups):
            ret.append(self.decoders[i](out[:, i]))
        ret = torch.stack(ret, 1).reshape(N, T, self.hp.n_cluster_groups, -1)
        return ret

    def infer(self, cond, gt=None):
        #cond: N, 1, C
        #gt: N, (1~3) or None
        i = 0
        cond = self.linear(cond)
        if gt is not None:
            i = gt.size(1)
            inp = [cond]
            for j, gt_split in enumerate(torch.split(gt, 1, dim=1)):
                gt_o = self.embedding[j](gt_split) #N, 1, C
                inp.append(gt_o)
            cond = torch.cat(inp, 1) #N, (1~4), C
        cond = self.layer_norm(cond)
        ret, _, _, _ = self.model(cond, memory=None, tgt_mask=self.tgt_mask[:i+1, :i+1])
        return self.decoders[i](ret[:, i]) #1, n_code


class Transducer(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        assert hp.hidden_size % hp.n_cluster_groups == 0
        per_embedding = hp.hidden_size // hp.n_cluster_groups
        #End Start Repetition (Although End Token should not be encoded)
        self.n_decoder_codes = hp.n_codes + 3 if self.hp.use_repetition_token else hp.n_codes + 2
        self.embeddings = nn.ModuleList([nn.Embedding(self.n_decoder_codes, per_embedding) for i in range(hp.n_cluster_groups)])
        self.decoder = ARCodeTransformer(hp, self.n_decoder_codes)

    def start_token(self, device):
        ret = []
        for embed in self.embeddings:
            ret.append(embed(torch.zeros((1, 1), device=device, dtype=torch.long) + self.hp.n_codes + 1))
        ret = torch.cat(ret, 2) # 1, 1, C
        return ret

    def truncate_to_end_token(self, x):
        #x: 1, T, 4
        mask = torch.any(x == self.hp.n_codes, -1) #1, T
        mask = mask.float().cumsum(-1) == 0
        ret = x[mask].unsqueeze(0)
        return ret

    def is_end_token(self, x):
        #All clusters goes to end -> end
        #x: 1, 1, 4
        ret = torch.any(x.squeeze() == self.hp.n_codes).item()
        return ret

    def is_end_token_batch(self, x):
        #All clusters goes to end -> end
        #x: N, 1, 4
        ret = torch.any(x.squeeze() == self.hp.n_codes, dim=-1).squeeze(-1)
        return ret #N,

    def is_end_token_beam(self, x):
        #All clusters and beams goes to end -> end
        #x: bs, T, 4
        ret = torch.any(x.view(x.size(0), -1) == self.hp.n_codes, -1) #bs,
        ret = torch.all(ret).item()
        return ret

    def encode(self, x):
        #x: N, T, 4
        x = torch.split(x, 1, 2)
        ret = []
        for q, embed in zip(x, self.embeddings):
            q = embed(q.squeeze(-1))
            ret.append(q)
        ret = torch.cat(ret, -1)
        return ret

    def decode(self, c, gt):
        return self.decoder(c, gt)

