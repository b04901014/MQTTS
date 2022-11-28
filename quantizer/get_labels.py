from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE, mel_spectrogram, load_wav
from librosa.util import normalize
from models import Generator, Encoder, Quantizer
from tqdm import tqdm
from pyannote.audio import Inference
import json

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    encoder = Encoder(h).to(device)
    quantizer = Quantizer(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    encoder.load_state_dict(state_dict_g['encoder'])
    quantizer.load_state_dict(state_dict_g['quantizer'])

    with open(a.input_json, 'r') as f:
        t_file = json.load(f)
    filelist = [os.path.join(a.input_wav_dir, l) for l in t_file]


    encoder.eval()
    quantizer.eval()
    encoder.remove_weight_norm()
    with torch.no_grad():
        for filname in tqdm(list(t_file.keys())):
            fname = os.path.join(a.input_wav_dir, filname)
            audio, sampling_rate = load_wav(fname)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            x = torch.FloatTensor(audio).to(device).unsqueeze(0)
            c = encoder(x.unsqueeze(1))
            q, loss_q, c = quantizer(c)
            t_file[filname]['quantization'] = [cc.tolist() for cc in c]

    with open(a.output_json, 'w') as f:
        json.dump(t_file, f)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='../datasets/train.json')
    parser.add_argument('--input_wav_dir', default='../datasets/audios')
    parser.add_argument('--output_json', default='../datasets/train_q.json')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print (device)

    inference(a)


if __name__ == '__main__':
    main()

