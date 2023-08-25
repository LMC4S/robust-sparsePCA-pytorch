# -*- coding: utf-8 -*-

import torch
from spca_gan import SpcaGAN
from config import Config

# Create a null SpcaGAN instance with default config if not run as a script.  ```from train import model```
# config is parsed from command line input if run as a script.
assert torch.cuda.is_available(), 'A CUDA GPU is required.'
config = Config().parse()
model = SpcaGAN(config)

if __name__ == '__main__':
    print(config.msg)

    # Train the model, write tensorboard event files, evaluate errors and write to csv file.
    model.train()
    model.evaluate(write_to_csv=True)