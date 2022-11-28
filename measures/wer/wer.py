from pathlib import Path
import jiwer
import json
import argparse
from tqdm import tqdm

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()
])


parser = argparse.ArgumentParser()
parser.add_argument('--cachepath', type=str, required=True)
args = parser.parse_args()

with open(args.cachepath, 'r') as f:
    data = json.load(f)

ref, target = [], []
for a, b in tqdm(data):
    ref.append(a)
    target.append(b)

measures = jiwer.compute_measures(ref, target,
                truth_transform=transformation,
                hypothesis_transform=transformation)
print (measures['wer'])
print (measures['deletions'] / (measures['deletions'] + measures['substitutions'] + measures['hits']))
