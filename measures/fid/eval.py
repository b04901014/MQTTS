from scipy.special import softmax
import audeer
import audonnx
import numpy as np
import json
import os
import soundfile as sf
from tqdm import tqdm

url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')

archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root, device='cuda')

sampling_rate = 16000

def eval_aud(aud):
    return model(aud, sampling_rate)['hidden_states'][0]

metapath = '../tts_fid_batch.json'
audiodir = 'samples/fid_{...}'
with open(metapath, 'r') as f:
    meta = json.load(f)
buff_gt = []
n_examples = len(list(os.listdir(audiodir)))
for i in tqdm(list(range(n_examples))):
    fname = f"{audiodir}/sentence-{i+1}-1.wav"
    audio_gt, sr = sf.read(fname)
    gt_vad = eval_aud(audio_gt.astype(np.float32))
    buff_gt.append(gt_vad)
buff_gt = np.array(buff_gt)
np.save('cache/{...}', buff_gt)
