# MQTTS
 - Official implementation for the paper [TODO]().
 - Audio samples (40 each system) can be accessed at [here](https://cmu.box.com/s/ktbk9pi04e2z1dlyepkkw69xcu9w91dj).
 - Quick demo can be accessed [TODO]().
## Setup the environment
1. Setup conda environment:
```
conda create --name mqtts python=3.9
conda activate mqtts
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```
(Update) You may need to create an access token to use the speaker embedding of pyannote as they updated their policy.
If that's the case follow the [pyannote repo](https://github.com/pyannote/pyannote-audio) and change every `Inference("pyannote/embedding", window="whole")` accordingly.

2. Download the pretrained phonemizer checkpoint
```
wget https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt
```

## Preprocess the dataset
1. Get the GigaSpeech dataset from the [official repo](https://github.com/SpeechColab/GigaSpeech)
2. Install [FFmpeg](https://ffmpeg.org), then
```
conda install ffmpeg=4.3=hf484d3e_0
conda update ffmpeg
```
3. Run python script
```
python preprocess.py --giga_speech_dir GIGASPEECH --outputdir datasets 
```

## Train the quantizer and inference
1. Train
```
cd quantizer/
python train.py --input_wavs_dir ../datasets/audios \
                --input_training_file ../datasets/training.txt \
                --input_validation_file ../datasets/validation.txt \
                --checkpoint_path ./checkpoints \
                --config config.json
```

2. Inference to get codes for training the second stage
```
python get_labels.py --input_json ../datasets/train.json \
                     --input_wav_dir ../datasets/audios \
                     --output_json ../datasets/train_q.json \
                     --checkpoint_file ./checkpoints/g_{training_steps}
python get_labels.py --input_json ../datasets/dev.json \
                     --input_wav_dir ../datasets/audios \
                     --output_json ../datasets/dev_q.json \
                     --checkpoint_file ./checkpoints/g_{training_steps}
```

## Train the transformer (below an example for the 100M version)
```
cd ..
mkdir ckpt
python train.py \
     --distributed \
     --saving_path ckpt/ \
     --sampledir logs/ \
     --vocoder_config_path quantizer/checkpoints/config.json \
     --vocoder_ckpt_path quantizer/checkpoints/g_{training_steps} \
     --datadir datasets/audios \
     --metapath datasets/train_q.json \
     --val_metapath datasets/dev_q.json \
     --use_repetition_token \
     --ar_layer 4 \
     --ar_ffd_size 1024 \
     --ar_hidden_size 256 \
     --ar_nheads 4 \
     --speaker_embed_dropout 0.05 \
     --enc_nlayers 6 \
     --dec_nlayers 6 \
     --ffd_size 3072 \
     --hidden_size 768 \
     --nheads 12 \
     --batch_size 200 \
     --precision bf16 \
     --training_step 800000 \
     --layer_norm_eps 1e-05
```
You can view the progress using:
```
tensorboard --logdir logs/
```

## Run batched inference (You'll have to change `speaker_to_text.json`, it's just an example.)
```
mkdir infer_samples
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --phonemizer_dict_path en_us_cmudict_forward.pt \
    --model_path ckpt/last.ckpt \
    --config_path ckpt/config.json \
    --input_path speaker_to_text.json \
    --outputdir infer_samples \
    --batch_size {batch_size} \
    --top_p 0.8 \
    --min_top_k 2 \
    --max_output_length {Maximum Output Frames to prevent infinite loop} \
    --phone_context_window 3 \
    --clean_speech_prior
```

### Pretrained checkpoints

1. Quantizer (put it under `quantizer/checkpoints/`):
```
wget https://anonfiles.com/Tf52ua4dy8/g_00600000
```

2. Transformer (100M version) (put it under `ckpt/`):
```
wget https://anonfiles.com/o6C1u747y6/last_ckpt
```
