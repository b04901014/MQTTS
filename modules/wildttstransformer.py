import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers import TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder, CrossAttnOnlyLayer, AlibiPostionEmbedding
from .transducer import Transducer
import numpy as np
import statistics

class TTSDecoder(nn.Module):
    def __init__(self, hp, phoneset_size):
        super().__init__()
        self.hp = hp
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                [TransformerEncoderLayer(hp) for i in range(hp.enc_nlayers)]
            )
        )
        self.decoder = TransformerDecoder(
            nn.ModuleList(
                [TransformerDecoderLayer(hp, with_cross_attention=False) for i in range(hp.dec_nlayers)]
            )
        )
        self.aligner = CrossAttnOnlyLayer(hp)
        self.layer_norm_phone = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.layer_norm_spkr = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.transducer = Transducer(hp)
        self.alibi = AlibiPostionEmbedding(hp.nheads, 10000)
        self.layer_norm = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.tgt_mask = (torch.tril(torch.ones(10000, 10000), diagonal=0) == 0)

    def forward(self, q, phone, spkr, q_mask, phone_mask):
        #Fused phone + speaker
        ex_phone_mask = torch.cat([torch.zeros((spkr.size(0), 1), device=spkr.device, dtype=torch.bool),
                                   phone_mask], 1) if phone_mask is not None else None
        spkr = self.layer_norm_spkr(spkr.unsqueeze(1))
        phone = self.layer_norm_phone(phone)
        phone = torch.cat([spkr, phone], 1)
        phone_alibi = self.alibi(phone)
        phone_alibi[:, 0] = 0
        phone_alibi[:, :, 0] = 0
        phone, enc_attn = self.encoder(phone, mask=None, attn_bias=phone_alibi, src_key_padding_mask=ex_phone_mask)
        phone = phone[:, 1:]
        #Run decoder
        q_mask = torch.cat([torch.zeros((spkr.size(0), 1), device=spkr.device, dtype=torch.bool),
                            q_mask], 1) if q_mask is not None else None
        q_input = q
        q = self.transducer.encode(q)
        q = self.layer_norm(q)
        q = torch.cat([spkr, q], 1)
        tgt_len = q.size(1)
        tgt_mask = self.tgt_mask[: tgt_len, : tgt_len].to(q.device)
        audio_alibi = self.alibi(q)
        audio_alibi[:, 0] = 0
        audio_alibi[:, :, 0] = 0
        output, _, dec_attn, _ = self.decoder(q, memory=None,
                                              tgt_mask=tgt_mask,
                                              attn_bias=audio_alibi,
                                              tgt_key_padding_mask=q_mask,
                                              memory_key_padding_mask=None)
        output, alignment = self.aligner(output, phone, tgt_mask=tgt_mask,
                                         tgt_key_padding_mask=q_mask, memory_key_padding_mask=phone_mask)
        audio_output = output[:, 1:]
        audio_output = self.transducer.decode(audio_output, q_input)
        return {
            'logits': audio_output,
            'alignment': alignment,
            'decoder_attention': dec_attn,
            'encoder_attention': enc_attn
        }

    def encode_phone(self, phone, spkr, phone_mask):
        phone = self.layer_norm_phone(phone)
        phone = torch.cat([spkr, phone], 1)
        ex_phone_mask = torch.cat([torch.zeros((spkr.size(0), 1), device=spkr.device, dtype=torch.bool), phone_mask], 1)
        phone_alibi = self.alibi(phone)
        phone_alibi[:, 0] = 0
        phone_alibi[:, :, 0] = 0
        phone, enc_attn = self.encoder(phone, mask=None, attn_bias=phone_alibi, src_key_padding_mask=ex_phone_mask)
        phone = phone[:, 1:]
        return phone

    def inference_topkp_sampling_batch(self, phone, spkr, phone_mask, prior=None, output_alignment=False):
        batch_size = phone.size(0)
        final_outputs = [0 for _ in range(batch_size)]
        spkr = self.layer_norm_spkr(spkr.unsqueeze(1))
        inp = self.layer_norm(self.transducer.start_token(phone.device)) #1, 1, C
        inp = inp.expand(batch_size, -1, -1) #N, 1, C
        inp = torch.cat([spkr, inp], 1)
        prior_size = 0
        if prior is not None:
            prior = self.transducer.encode(prior)
            prior = self.layer_norm(prior)
            prior_size = prior.size(1)
            inp = torch.cat([inp, prior], 1)
        phone = self.encode_phone(phone, spkr, phone_mask)
        tgt_mask = self.tgt_mask[:inp.size(1), :inp.size(1)].to(inp.device)
        inps = inp
        #Decode
        past_kvs1, past_kv_cross, past_kvs2, clusters = None, None, None, torch.empty([batch_size, 0, self.hp.n_cluster_groups], device=phone.device, dtype=torch.long)
        audio_alibi = self.alibi(inp)
        audio_alibi[:, 0] = 0
        audio_alibi[:, :, 0] = 0
        back_map = torch.zeros([batch_size, 1], device=phone.device, dtype=torch.long)
        length_counter = torch.zeros([batch_size], device=phone.device, dtype=torch.long)
        real_phone_lengths = (~phone_mask).long().sum(-1) #N,
        if output_alignment:
            assert batch_size == 1, "Now only support output alignment for bs = 1 for debugging issues..."
            alignment = torch.zeros((1, self.hp.max_output_length, self.hp.max_output_length), device=phone.device)
        for i in range(self.hp.max_output_length):
            cond, _, _, new_1 = self.decoder(inp, memory=None, attn_bias=audio_alibi, tgt_mask=tgt_mask, past_kvs=past_kvs1)
            #Only feed in the current frame and the next frame attending!
            t_length, c_length = phone.size(1), phone.size(2) # T, C
            selected_phone = phone.reshape(-1, c_length) #N*T, C
            index_map = torch.arange(self.hp.phone_context_window, device=phone.device)
            index_map = back_map[:, -1:] + index_map.repeat(batch_size, 1)
            add = torch.arange(batch_size, device=index_map.device).unsqueeze(1) #N, 1
            index_map = index_map + add * t_length
            index_map = index_map.reshape(-1) #N * 3
            selected_phone = selected_phone[index_map].reshape(batch_size, self.hp.phone_context_window, c_length) #N*3, C
            #Mask for the starting phones
            phone_mask = torch.arange(self.hp.phone_context_window, device=phone.device).repeat(batch_size, 1)
            phone_mask = (phone_mask <= (back_map[:, -1:] + 1).expand(-1, self.hp.phone_context_window))
            phone_mask = ~phone_mask
            cond, _align = self.aligner(cond, selected_phone, tgt_mask=tgt_mask, memory_key_padding_mask=phone_mask)
            cond = cond[:, -1].unsqueeze(1) #N, 1, C
            #Run sub-decoder inference
            output = []
            for j in range(self.hp.n_cluster_groups):
                q_input = torch.cat(output, 1) if j else None
                logit = self.transducer.decoder.infer(cond, q_input) #N, n_codes
                #Block Start Token
                logit[:, self.hp.n_codes + 1] = -float("Inf")
                #Don't output stop token if alignment not near end
                logit_tmp = logit[back_map[:, -1] < (real_phone_lengths - 2)]
                logit_tmp[:, self.hp.n_codes] = -float("Inf")
                logit[back_map[:, -1] < (real_phone_lengths - 2)] = logit_tmp
                #Repetition penalty
                if self.hp.use_repetition_token and self.hp.repetition_penalty != 1.0:
                    logit[:, self.hp.n_codes + 2] /= self.hp.repetition_penalty
                if self.hp.use_repetition_gating:
                    logit[:, self.hp.n_codes + 2] = torch.min(torch.max(logit[:, :self.hp.n_codes]), logit[:, self.hp.n_codes + 2])
                #Top_p
                if self.hp.top_p < 1.0 and self.hp.top_p > 0.0:
                    sorted_logits, sorted_idxs = torch.sort(logit, descending=True)
                    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    additional_prob = (self.hp.length_penalty_max_prob - self.hp.top_p) * (length_counter / self.hp.length_penalty_max_length)
                    idx_to_remove = cum_probs > (self.hp.top_p + additional_prob).unsqueeze(-1)
                    idx_to_remove[:, :self.hp.min_top_k] = False
                    idx_to_remove = idx_to_remove.scatter(1, sorted_idxs, idx_to_remove)
                    logit[idx_to_remove] = -float("Inf")
                #Sampling
                probs = torch.softmax(logit / self.hp.sampling_temperature, dim=-1)
                idx = torch.multinomial(probs, 1) #N, 1
                #If is repetition token
                if self.hp.use_repetition_token:
                    if clusters.size(1) == 0: #First token, random choice
                        idx[idx==(self.hp.n_codes + 2)] = torch.randint(self.hp.n_codes, size=(1,), device=idx.device)
                    else:
                        idx[idx==(self.hp.n_codes + 2)] = clusters[:, -1:, j][idx==(self.hp.n_codes + 2)]
                output.append(idx)
            output = torch.cat(output, 1).unsqueeze(1) #N, 1, n_groups
            #Stop criterion
            stopping_streams = (back_map[:, -1] == (real_phone_lengths - self.hp.phone_context_window))
            stopping_streams = (stopping_streams & self.transducer.is_end_token_batch(output)) | (stopping_streams & (torch.argmax(_align[:, 0, -1], dim=-1) == self.hp.phone_context_window - 1)) #N,
            if i == self.hp.max_output_length - 1:
                stopping_streams[:] = True
            stopping_streams_idx = np.where(stopping_streams.detach().cpu().numpy())[0]
            num_stopped = stopping_streams.long().sum().item()
            if num_stopped > 0:
                stopped = clusters[stopping_streams]
                n_seats, stop_seats = 0, 0
                for n_s, seat in enumerate(final_outputs):
                    if type(seat) == int:
                        n_seats += 1
                        if n_seats - 1 in stopping_streams_idx:
#                            print (n_seats, stopping_streams_idx, stopped.size(), stop_seats)
                            final_outputs[n_s] = stopped[stop_seats]
                            stop_seats += 1
            n_remained = sum([int(type(x) == int) for x in final_outputs])
            if n_remained == 0:
                break
            #Trim batches
            batch_size = batch_size - num_stopped
            output = output[~stopping_streams]
            phone = phone[~stopping_streams]
            real_phone_lengths = real_phone_lengths[~stopping_streams]
            clusters = clusters[~stopping_streams]
            back_map = back_map[~stopping_streams]
            length_counter = length_counter[~stopping_streams]
            _align = _align[~stopping_streams]
            news = [inps] + new_1
            inps = inps[~stopping_streams]
            for layer in range(len(news)):
                news[layer] = news[layer][~stopping_streams]
            if past_kvs1 is not None:
                for layer in range(len(past_kvs1)):
                    past_kvs1[layer] = past_kvs1[layer][~stopping_streams]

            #Update args
            tgt_mask = self.tgt_mask[i+3+prior_size, :i+3+prior_size].to(phone.device).unsqueeze(0)
            audio_alibi = self.alibi(tgt_mask)[:, -1].unsqueeze(1)
            audio_alibi[:, :, 0] = 0
            if output_alignment:
                alignment[:, i, back_map[0, -1]: back_map[0, -1]+self.hp.phone_context_window] = _align[:, 0, -1].unsqueeze(0)
            next_idx = (_align[:, 0, -1, 0] < (1 / self.hp.phone_context_window)).long()
            next_idx[length_counter >= self.hp.length_penalty_max_length] = 1
            new_bk = torch.minimum(back_map[:, -1] + next_idx, real_phone_lengths - self.hp.phone_context_window)
            back_map = torch.cat([back_map, new_bk.unsqueeze(1)], 1)
            length_counter[next_idx == 0] += 1
            length_counter[next_idx != 0] = 0
            if i == 0:
                past_kvs1 = news[:self.hp.dec_nlayers]
            else:
                news = [x[:, -1:] for x in news]
                for ii, (p, n) in enumerate(zip(past_kvs1, news[:self.hp.dec_nlayers])):
                    past_kvs1[ii] = torch.cat([p, n], 1)

            inp = self.transducer.encode(output)
            inp = self.layer_norm(inp)
            inps = torch.cat([inps, inp], 1)
            clusters = torch.cat([clusters, output], 1) #N, T, 4
        if output_alignment:
            return final_outputs, alignment[:, :i, :phone.size(1)]
        return final_outputs
