import soundfile as sf
import argparse
import soundfile as sf
from pathlib import Path
import numpy as np
from pyannote.audio import Inference
import json
from librosa.util import normalize
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--metapath', type=str, required=True)
args = parser.parse_args()

with open(args.metapath, 'r') as f:
    meta = json.load(f)

spkr_embedding_model = Inference("pyannote/embedding", window="whole", device='cuda')

def spkr_embed(path):
    ref, sampling_rate = sf.read(path)
    if len(ref) > 10000:
        ref = normalize(ref) * 0.95
        ref = spkr_embedding_model({'waveform': torch.FloatTensor(ref).unsqueeze(0), 'sample_rate': sampling_rate})
        return ref

scores = []

for i, (ref, _) in enumerate(meta):
    ref = spkr_embed(ref)
    target = os.path.join(args.datadir, f'sentence-{i+1}-1.wav')
    target = spkr_embed(target)
    if target is not None:
        score = np.dot(ref, target) / (np.linalg.norm(ref) * np.linalg.norm(target))
        scores.append(score)

similarity = np.mean(scores)
print (similarity)
