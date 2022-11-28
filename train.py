from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from trainer import Wav2TTS
from pytorch_lightning.plugins import DDPPlugin
import argparse
import json
import os

parser = argparse.ArgumentParser()

#Paths
parser.add_argument('--saving_path', type=str, default='./ckpt')
parser.add_argument('--resume_checkpoint', type=str, default=None)
parser.add_argument('--vocoder_config_path', type=str, required=True)
parser.add_argument('--vocoder_ckpt_path', type=str, required=True)
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--metapath', type=str, required=True)
parser.add_argument('--val_metapath', type=str, required=True)
parser.add_argument('--sampledir', type=str, default='./logs')
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--speaker_embedding_dir', type=str, default=None)

#Optimizer
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=float, default=150)
parser.add_argument('--train_bucket_size', type=int, default=8192)
parser.add_argument('--training_step', type=int, default=800000)
parser.add_argument('--optim_flat_percent', type=float, default=0.0)
parser.add_argument('--warmup_step', type=int, default=50)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.98)

#Architecture
parser.add_argument('--ffd_size', type=int, default=3072)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--enc_nlayers', type=int, default=4)
parser.add_argument('--dec_nlayers', type=int, default=2)
parser.add_argument('--nheads', type=int, default=12)
parser.add_argument('--ar_layer', type=int, default=1)
parser.add_argument('--ar_ffd_size', type=int, default=1024)
parser.add_argument('--ar_hidden_size', type=int, default=256)
parser.add_argument('--ar_nheads', type=int, default=4)
parser.add_argument('--aligner_softmax_temp', type=float, default=1.0)
parser.add_argument('--layer_norm_eps', type=float, default=1e-5)

#Dropout
parser.add_argument('--speaker_embed_dropout', type=float, default=0.0)
parser.add_argument('--label_smoothing', type=float, default=0.0)

#Trainer
parser.add_argument('--val_check_interval', type=int, default=5000)
parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
parser.add_argument('--precision', type=str, choices=['16', '32', "bf16"], default=32)
parser.add_argument('--nworkers', type=int, default=16)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--accelerator', type=str, default='ddp')
parser.add_argument('--version', type=int, default=None)
parser.add_argument('--accumulate_grad_batches', type=int, default=1)

#Sampling
parser.add_argument('--use_repetition_token', action='store_true')
parser.add_argument('--use_repetition_gating', action='store_true')
parser.add_argument('--repetition_penalty', type=float, default=1.0)
parser.add_argument('--sampling_temperature', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=-1)
parser.add_argument('--min_top_k', type=int, default=3)
parser.add_argument('--top_p', type=float, default=0.7)
parser.add_argument('--sample_num', type=int, default=4)
parser.add_argument('--length_penalty_max_length', type=int, default=15000)
parser.add_argument('--length_penalty_max_prob', type=float, default=0.95)
parser.add_argument('--max_input_length', type=int, default=2048)
parser.add_argument('--max_output_length', type=int, default=1500)
parser.add_argument('--phone_context_window', type=int, default=3)

#Data
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--n_codes', type=int, default=160)
parser.add_argument('--n_cluster_groups', type=int, default=4)


args = parser.parse_args()

with open(os.path.join(args.saving_path, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

fname_prefix = f''

if args.accelerator == 'ddp':
    args.accelerator = DDPPlugin(find_unused_parameters=False)

checkpoint_callback = ModelCheckpoint(
    dirpath=args.saving_path,
    filename=(fname_prefix+'{epoch}-{step}'),
    every_n_train_steps=(None if args.val_check_interval == 1.0 else args.val_check_interval),
    every_n_epochs=(None if args.check_val_every_n_epoch == 1 else args.check_val_every_n_epoch),
    verbose=True,
    save_last=True
)

logger = TensorBoardLogger(args.sampledir, name="VQ-TTS", version=args.version)

wrapper = Trainer(
    precision=args.precision,
    amp_backend='native',
    callbacks=[checkpoint_callback],
    resume_from_checkpoint=args.resume_checkpoint,
    val_check_interval=args.val_check_interval,
    num_sanity_val_steps=0,
    max_steps=args.training_step,
    gpus=(-1 if args.distributed else 1),
    strategy=(args.accelerator if args.distributed else None),
    replace_sampler_ddp=False,
    accumulate_grad_batches=args.accumulate_grad_batches,
    logger=logger,
    check_val_every_n_epoch=args.check_val_every_n_epoch
)
model = Wav2TTS(args)
wrapper.fit(model)
