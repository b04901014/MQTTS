import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.QuantizeDataset import QuantizeDataset, QuantizeDatasetVal
from data.sampler import RandomBucketSampler
from modules.wildttstransformer import TTSDecoder
from modules.transformers import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer
from modules.vocoder import Vocoder
from torch.utils import data
import pytorch_lightning.core.lightning as pl
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class Wav2TTS(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.data = QuantizeDataset(hp, hp.metapath)
        self.val_data = QuantizeDatasetVal(hp, hp.val_metapath)
        self.TTSdecoder = TTSDecoder(hp, len(self.data.phoneset))
        self.n_decode_codes = self.TTSdecoder.transducer.n_decoder_codes
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=self.hp.label_smoothing)
        self.phone_embedding = nn.Embedding(len(self.data.phoneset), hp.hidden_size, padding_idx=self.data.phoneset.index('<pad>'))
        self.spkr_linear = nn.Linear(512, hp.hidden_size)
        if self.hp.pretrained_path:
            self.load()
        else:
            self.apply(self.init_weights)
        self.vocoder = Vocoder(hp.vocoder_config_path, hp.vocoder_ckpt_path)
        self.vocoder.eval()
        self.vocoder.generator.remove_weight_norm()
        for param in self.vocoder.parameters():
            param.requires_grad = False

    def load(self):
        state_dict = torch.load(self.hp.pretrained_path)['state_dict']
        self.load_state_dict(state_dict, strict=False)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module._fill_padding_idx_with_zero()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def train_dataloader(self):
        length = self.data.lengths
        sampler = RandomBucketSampler(self.hp.train_bucket_size, length, self.hp.batch_size, drop_last=True, distributed=self.hp.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank)
        dataset = data.DataLoader(self.data,
                                  num_workers=self.hp.nworkers,
                                  batch_sampler=sampler,
                                  collate_fn=self.data.seqCollate)
        return dataset

    def val_dataloader(self):
        dataset = data.DataLoader(self.val_data,
                                  num_workers=self.hp.nworkers,
                                  shuffle=False)
        return dataset

    def configure_optimizers(self):
        optimizer_adam = optim.Adam(self.parameters(), lr=self.hp.lr, betas=(self.hp.adam_beta1, self.hp.adam_beta2))
        #Learning rate scheduler
        num_training_steps = self.hp.training_step
        num_warmup_steps = self.hp.warmup_step
        num_flat_steps = int(self.hp.optim_flat_percent * num_training_steps)
        def lambda_lr(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step < (num_warmup_steps + num_flat_steps):
                return 1.0
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (num_warmup_steps + num_flat_steps)))
            )
        scheduler_adam = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer_adam, lambda_lr),
            'interval': 'step'
        }
        return [optimizer_adam], [scheduler_adam]

    def training_step(self, batch, batch_idx):
        #Deal with speaker embedding
        speaker_embedding = F.normalize(batch['speaker'], dim=-1)
        speaker_embedding = self.spkr_linear(F.dropout(speaker_embedding, self.hp.speaker_embed_dropout))
        #Deal with phone segments
        phone_features = self.phone_embedding(batch['phone'])
        #Run decoder
        recons_segments = self.TTSdecoder(batch['tts_quantize_input'], phone_features, speaker_embedding,
                                          batch['quantize_mask'], batch['phone_mask'])
        target = recons_segments['logits'][~batch['quantize_mask']].view(-1, self.n_decode_codes)
        labels = batch['tts_quantize_output'][~batch['quantize_mask']].view(-1)
        loss = self.cross_entropy(target, labels)
        acc = (target.argmax(-1) == labels).float().mean()
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        #For the first half samples, and random choose the rest half
        start_point, half = 4, self.hp.sample_num // 2
        if self.hp.sample_num > 0:
            self.sample_idxs = list(range(start_point, start_point + half)) + \
                np.random.randint(low=start_point + half, high=len(self.val_data), size=self.hp.sample_num//2).tolist()
        else:
            self.sample_idxs = []

    def validation_step(self, batch, batch_idx):
        #Batch size = 1
        spkr, q_s, q_e, phone, ground_truth = batch
        norm_spkr = F.normalize(spkr, dim=-1)
        spkr = self.spkr_linear(norm_spkr)
        phone_features = self.phone_embedding(phone)
        recons_segments = self.TTSdecoder(q_s, phone_features, spkr, None, None)
        target = recons_segments['logits'].view(-1, self.n_decode_codes)
        labels = q_e.view(-1)
        loss = self.cross_entropy(target, labels)
        acc = (target.argmax(-1) == labels).float().mean()
        self.log("val/loss", loss, on_epoch=True, logger=True)
        self.log("val/acc", acc, on_epoch=True, logger=True)

        #Run inference with bs = 1
        if batch_idx in self.sample_idxs:
            batch_idx = self.sample_idxs.index(batch_idx)
            phone_mask = torch.full((phone_features.size(0), phone_features.size(1)), False, dtype=torch.bool, device=phone_features.device)
            synthetic, infer_attn = self.TTSdecoder.inference_topkp_sampling_batch(phone_features, spkr, phone_mask, prior=None, output_alignment=True)
            synthetic = synthetic[0].unsqueeze(0)
            synthetic = self.vocoder(synthetic, norm_spkr).float()
            #Reconstructed Audio with vocoder
            reconstructed_gt = self.vocoder(q_s[:, 1:], norm_spkr).float()
            #Write files
            sw = self.logger.experiment
            sw.add_audio(f'generated/{batch_idx}', synthetic, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'vocoder-reconstructed/{batch_idx}', reconstructed_gt, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'groundtruth/{batch_idx}', ground_truth[0], self.global_step, self.hp.sample_rate)

            #Plot attentions
            self.plot_attn(recons_segments['encoder_attention'], f'enc-attn/{batch_idx}', (10, 10))
            self.plot_attn(recons_segments['decoder_attention'], f'dec-attn/{batch_idx}', (10, 10))
            self.plot_attn([recons_segments['alignment']], f'train-alignment/{batch_idx}', (10, 10))
            self.plot_attn([infer_attn.unsqueeze(0)], f'infer-alignment/{batch_idx}', (10, 10))

    def plot_attn(self, attns, prefix, figsize):
        nheads = attns[0].size(1)
        fig, axs = plt.subplots(len(attns), nheads, constrained_layout=True, figsize=figsize)
        if len(attns) == 1 and nheads == 1:
            axs = [[axs]]
        elif len(attns) == 1 or nheads == 1:
            axs = [axs]
        for i, attn in enumerate(attns): #Each layers
            attn = attn.float().cpu().numpy()
            for j, head_attn in enumerate(attn[0]):
                axs[i][j].matshow(head_attn, aspect="auto", origin="lower", interpolation='none')
                if i != 0 or j != 0:
                    axs[i][j].axis('off')
        self.logger.experiment.add_figure(prefix, fig, self.global_step)
        plt.close()
