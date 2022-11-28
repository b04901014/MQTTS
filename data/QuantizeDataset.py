import os
from torch.utils import data
import torch
import json
import numpy as np
import soundfile as sf
import random
from pathlib import Path
from librosa.util import normalize
from pyannote.audio import Inference

import torch.nn.functional as F

def random_crop(x, maxseqlen):
    if x.shape[0] >= maxseqlen:
        offset = random.randrange(x.shape[0] - maxseqlen + 1)
        x = x[offset: offset + maxseqlen]
    else:
        offset = 0
    return x, offset

def dynamic_range_compression(x, C=0.3, M=6.5, clip_val=1e-5):
    return (np.log(np.clip(x, a_min=clip_val, a_max=None)) + M) * C

def dynamic_range_decompression(x, C=0.3, M=6.5):
    return np.exp(x / C - M)

class QuantizeDataset(data.Dataset):
    def __init__(self, hp, metapath):
        self.hp = hp
        print (f'Loading metadata in {metapath}...')
        with open(metapath, 'r') as f:
            self.text = json.load(f) #{name: {text:, phoneme:, ..., duration: }}
        self.datasetbase = [x for x in self.text.keys()]
        self.dataset = [os.path.join(self.hp.datadir, x) for x in self.datasetbase]
        self.phoneset = ['<pad>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', ',', '.']
        print (self.phoneset)
        if self.hp.speaker_embedding_dir is None:
            self.spkr_embedding = Inference("pyannote/embedding", window="whole")

        #Print statistics:
        l = len(self.dataset)
        print (f'Total {l} examples')

        self.lengths = [float(v['duration']) for v in self.text.values()]
        avglen = sum(self.lengths) / len(self.lengths)
        maxlen = max(self.lengths)
        minlen = min(self.lengths)
        print (f"Average duration of audio: {avglen} sec, Maximum duration: {maxlen} sec, Minimum duration: {minlen} sec")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        dataname = self.dataset[i]
        _name = self.datasetbase[i]
        metadata = self.text[_name]
        #To synthesized phoneme sequence
        phonemes = [self.phoneset.index(ph) for ph in metadata['phoneme'].split() if ph in self.phoneset]

        if self.hp.speaker_embedding_dir is None:
            audio, sampling_rate = sf.read(dataname)
            audio = normalize(audio) * 0.95
            speaker_embedding = self.spkr_embedding({'waveform': torch.FloatTensor(audio).unsqueeze(0), 'sample_rate': self.hp.sample_rate})
        else:
            speaker_embedding = os.path.join(self.hp.speaker_embedding_dir, os.path.splitext(_name)[0] + '.npy')
            speaker_embedding = np.load(speaker_embedding).astype(np.float32)

        #Ground truth for TTS system
        quantization = np.array(metadata['quantization']).T # ..., 4
        #Add start token, end token
        start, end = np.full((1, self.hp.n_cluster_groups), self.hp.n_codes + 1, dtype=np.int16), np.full((1, self.hp.n_cluster_groups), self.hp.n_codes, dtype=np.int16)
        quantization_s = np.concatenate([start, quantization.copy()], 0)
        #Add repetition token if needed for ground truth "label"
        if self.hp.use_repetition_token:
            pad = np.full((1, self.hp.n_cluster_groups), -100, dtype=np.int16)
            np_mask = np.diff(quantization, axis=0, prepend=pad)
            quantization[np_mask == 0] = self.hp.n_codes + 2
        quantization_e = np.concatenate([quantization, end], 0)
        return speaker_embedding, quantization_s, quantization_e, phonemes, dataname

    def seqCollate(self, batch):
        output = {
            'speaker': [],
            'phone': [],
            'phone_mask': [],
            'tts_quantize_input': [],
            'tts_quantize_output': [],
            'quantize_mask': [],
        }
        #Get the max length of everything
        max_len_q, max_phonelen = 0, 0
        for spkr, q_s, q_e, ph, _ in batch:
            if len(q_s) > max_len_q:
                max_len_q = len(q_s)
            if len(ph) > max_phonelen:
                max_phonelen = len(ph)
            output['speaker'].append(spkr)
        #Pad each element, create mask
        for _, qs, qe, phone, _ in batch:
            #Deal with phonemes
            phone_mask = np.array([False] * len(phone) + [True] * (max_phonelen - len(phone)))
            phone = np.pad(phone, [0, max_phonelen-len(phone)])
            #Deal with quantizations
            q_mask = np.array([False] * len(qs) + [True] * (max_len_q - len(qs)))
            qs = np.pad(qs, [[0, max_len_q-len(qs)], [0, 0]], constant_values=self.hp.n_codes)
            qe = np.pad(qe, [[0, max_len_q-len(qe)], [0, 0]], constant_values=self.hp.n_codes)
            #Aggregate
            output['phone'].append(phone)
            output['phone_mask'].append(phone_mask)
            output['tts_quantize_input'].append(qs)
            output['tts_quantize_output'].append(qe)
            output['quantize_mask'].append(q_mask)
        for k in output.keys():
            output[k] = np.array(output[k])
            if 'mask' in k:
                output[k] = torch.BoolTensor(output[k])
            elif k in ['phone', 'tts_quantize_input', 'tts_quantize_output']:
                output[k] = torch.LongTensor(output[k])
            else:
                output[k] = torch.FloatTensor(output[k])
        return output

class QuantizeDatasetVal(QuantizeDataset):
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        speaker_embedding, quantization_s, quantization_e, phonemes, dataname = super().__getitem__(i)
        audio, sampling_rate = sf.read(dataname)
        audio = normalize(audio) * 0.95
        return (
            torch.FloatTensor(speaker_embedding),
            torch.LongTensor(quantization_s),
            torch.LongTensor(quantization_e),
            torch.LongTensor(phonemes),
            torch.FloatTensor(audio)
        )
