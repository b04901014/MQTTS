import soundfile
import argparse
import soundfile as sf
from pathlib import Path
import jiwer
import numpy as np
from google.cloud import speech_v1 as speech
import io
import json

def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    client = speech.SpeechClient()
    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        model="video"
    )
    response = client.recognize(config=config, audio=audio)
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    output = []
    for result in response.results:
        output.append(result.alternatives[0].transcript)
    return ' '.join(output)

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()
])

results = []
caches = []

datadir = Path(args.datadir)
for fn in datadir.glob('*.wav'):
    with open(str(fn)[:-4] + '.txt', 'r') as f:
        ref = f.readlines()[0]
    target = transcribe_file(fn)
    wer = jiwer.wer(ref, target,
                    truth_transform=transformation,
                    hypothesis_transform=transformation)
    results.append(wer)
    caches.append([ref, target])
    print (ref)
    print (target)
    print (wer)
with open(args.output, 'w') as f:
    json.dump(caches, f)
print (f"Final WER: {np.mean(results)}")
