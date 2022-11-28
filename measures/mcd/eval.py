from pathlib import Path
import jiwer
import json
from tqdm import tqdm
import librosa
import pyworld
import pysptk
import soundfile as sf
import numpy as np
import math
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import multiprocessing

datadir = '../samples/mcd_{...}'
njobs = 16

def wav2mcep(wavfile, mcep_dim=23, mcep_alpha=0.42, n_fft=1024, n_shift=256):
    x, spr = sf.read(wavfile, dtype="int16")
    win = pysptk.sptk.hamming(n_fft)
    n_frame = (len(x) - n_fft) // n_shift + 1
    assert spr == 16000
    mcep = [
        pysptk.sptk.mcep(
                        x[n_shift * i : n_shift * i + n_fft] * win,
                        mcep_dim,
                        mcep_alpha,
                        eps=1e-8,
                        etype=1,
        )
                for i in range(n_frame)
    ]
    return np.stack(mcep)

def run(d):
    mcds = []
    for i, (tts_wav, gt_wav) in tqdm(enumerate(d)):
        tts_mcp = wav2mcep(tts_wav)
        gt_mcp = wav2mcep(gt_wav)
        ref_frame_no = len(gt_mcp)
        distance, path = fastdtw(tts_mcp, gt_mcp, dist=euclidean)
        path = np.array(path).T
        diff2sum = np.sum((tts_mcp[path[0]] - gt_mcp[path[1]]) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum))
        mcds.append(mcd)
    return mcds

if __name__ == '__main__':
    total_mcd = []
    with open('../dev_mcd.json', 'r') as f:
        data = json.load(f)
    gtwavs = [k for k, _ in data]
    ttswavs = [f'{datadir}/sentence-{i+1}-1.wav' for i in range(len(gtwavs))]
    data = list(zip(ttswavs, gtwavs))

    segment_size = int(len(data) / njobs + 1)
    segmented_data = [data[i*segment_size: (i+1)*segment_size] for i in range(njobs)]
    with multiprocessing.Pool(njobs) as p:
        out = p.map(run, segmented_data)
    for a in out:
        total_mcd += a
    print (np.mean(total_mcd), np.std(total_mcd))

