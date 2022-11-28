import os
import numpy as np
import soundfile as sf
import json
import random
from pathlib import Path
import sys
import subprocess
from tqdm import tqdm
import argparse
from dp.phonemizer import Phonemizer
import os
import pyloudnorm as pyln
from multiprocessing import Pool
import torchaudio
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--giga_speech_dir', type=str, required=True)
parser.add_argument('--outputdir', type=str, required=True)

args = parser.parse_args()

DATA_DIR = Path(args.giga_speech_dir) / Path('audio')
metadata_path = Path(args.giga_speech_dir) / Path('GigaSpeech.json')
phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_forward.pt')
output = {}

print ('Loading Labelfile...')
with open(str(metadata_path), 'r') as f:
    labels = json.load(f)['audios']
all_file_paths = [str(x) for x in DATA_DIR.rglob('*.opus')]

print ('Loading Filtered List...')
with open(os.path.join(args.outputdir, 'training.txt'), 'r') as f:
    training = [name.strip() for name in f.readlines()]
with open(os.path.join(args.outputdir, 'validation.txt'), 'r') as f:
    dev = [name.strip() for name in f.readlines()]

outputaudiodir = Path(args.outputdir) / Path('audios')
outputaudiodir.mkdir(exist_ok=True)
meter = pyln.Meter(16000)

def run(section):
    output_t, output_d = dict(), dict()
    for audiofile in tqdm(section[:20]):
        opus_path = os.path.join(args.giga_speech_dir, audiofile['path'])
        if opus_path in all_file_paths:
            #Check if one of the segments in file:
            start_run = False
            for k, sentence in enumerate(audiofile['segments']):
                sentence_sid = sentence['sid']
                if sentence_sid in training + dev:
                    start_run = True
            if not start_run:
                print ('sdfsdfsdfds')
                continue
            name = Path(opus_path).stem
            wav_path = os.path.join(outputaudiodir, name + '.wav')
            subprocess.run(f'ffmpeg -y -i {opus_path} -ac 1 -ar 16000 -acodec pcm_s16le {wav_path} -filter_threads 8', shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #Reload and segment the file
            audio, sr = torchaudio.load(wav_path)
            assert sr == 16000
            for k, sentence in enumerate(audiofile['segments']):
                sentence_sid = sentence['sid']
                if sentence_sid in training + dev:
                    begin_time = int(sentence['begin_time'] * sr)
                    end_time = int(sentence['end_time'] * sr)
                    sentence_path = os.path.join(outputaudiodir, f'{sentence_sid}.wav')
                    #Post-processing
                    seg_audio = audio[:, begin_time: end_time].numpy().mean(0)
                    loudness = meter.integrated_loudness(seg_audio)
                    seg_audio = pyln.normalize.loudness(seg_audio, loudness, -20.0)
                    fade_out = np.linspace(1.0, 0., 1600)
                    fade_in = np.linspace(0.0, 1.0, 1600)
                    seg_audio[:1600] *= fade_in
                    seg_audio[-1600:] *= fade_out
                    seg_audio = torch.FloatTensor(seg_audio).unsqueeze(0)
                    torchaudio.save(sentence_path, seg_audio, sample_rate=sr, format='wav', encoding='PCM_S', bits_per_sample=16)
                    #Text
                    text = sentence['text_tn'].lower().split()
                    for i, word in enumerate(text):
                        if word == '<comma>':
                            text[i] = ','
                        elif word == '<period>':
                            text[i] = '.'
                    text = [x for x in text if '<' not in x]
                    text = ' '.join(text)
                    phonemes = phonemizer(text, lang='en_us').replace('[', ' ').replace(']', ' ')
                    name = f'{sentence_sid}.wav'
                    if sentence_sid in training:
                        output_t[name] = {'text': text, 'duration': sentence['end_time'] - sentence['begin_time'], 'phoneme': phonemes}
                    else:
                        output_d[name] = {'text': text, 'duration': sentence['end_time'] - sentence['begin_time'], 'phoneme': phonemes}
            #Clean-up
            Path(wav_path).unlink()
    return output_t, output_d

if __name__ == '__main__':
    import random
    random.shuffle(labels)
    output_t, output_d = dict(), dict()
    print (all_file_paths[0])
    print (labels[0]['path'])
    output_t, output_d = run(labels)
    with open(os.path.join(args.outputdir, 'train.json'), 'w') as f:
        json.dump(output_t, f, indent=2)
    with open(os.path.join(args.outputdir, 'dev.json'), 'w') as f:
        json.dump(output_d, f, indent=2)
