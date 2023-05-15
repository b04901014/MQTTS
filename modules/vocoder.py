import torch.nn as nn
from quantizer.env import AttrDict
from quantizer.models import Generator, Quantizer, Encoder
import torch
import json

class Vocoder(nn.Module):
    def __init__(self, config_path, ckpt_path, with_encoder=False):
        super(Vocoder, self).__init__()
        ckpt = torch.load(ckpt_path)
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        self.generator.load_state_dict(ckpt['generator'])
        self.quantizer.load_state_dict(ckpt['quantizer'])
        if with_encoder:
            self.encoder = Encoder(self.h)
            self.encoder.load_state_dict(ckpt['encoder'])
    
    def process_chunk(self, input_waveform, chunk_idx, chunk_size, states):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, input_waveform.shape[-1])

        input_chunk = input_waveform[..., start_idx:end_idx]
        encoded, states = self.encoder(input_chunk, states)
        generated, states = self.generator(encoded, states)

        return generated, states

    def forward(self, x, spkr, chunk_size=None):
        if chunk_size is not None:
            # Initialize the states dictionary
            states = {}

            # Initialize an empty list to store the generated output chunks
            output_chunks = []

            # Calculate the number of chunks in the input waveform
            num_chunks = (x.shape[-1] + chunk_size - 1) // chunk_size

            # Process each chunk in the input waveform
            for chunk_idx in range(num_chunks):
                generated_chunk, states = process_chunk(self.encoder, self.generator, x, chunk_idx, chunk_size, states)
                output_chunks.append(generated_chunk)

            # Concatenate the generated output chunks to form the complete output
            return torch.cat(output_chunks, dim=-1)
        else:
            return self.generator(self.quantizer.embed(x), spkr)

    def encode(self, x):
        batch_size = x.size(0)
        c = self.encoder(x.unsqueeze(1))
        q, loss_q, c = self.quantizer(c)
        c = [code.reshape(batch_size, -1) for code in c]
        return torch.stack(c, -1) #N, T, 4
