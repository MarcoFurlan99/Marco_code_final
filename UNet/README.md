# U-Net

This is the code for the U-Net! (https://arxiv.org/abs/1505.04597)

An example of use is shown in "example.ipynb".

The original code is from https://github.com/milesial/Pytorch-UNet [^1].

[^1]: That code is in my opinion inefficient and a little messy. But if my code is not clear you should refer to that since I've built from it.

On top of it, I've added the following modifications:

- Reorganised it so it fits in a single, nice folder called "UNet"; to train and predict just call:

```bash
from UNet.train import train
```

```bash
from UNet.predict import predict
```

- Speeded up notably the unique mask values count operation (originally it converted a couple times from numpy array to PIL image for no reason).

- Removed useless lines of code, logging messages, and wandb statistics (not relevant for my uses).

- removed the parser/get args/if __name__=='__main__' because not fit for my purposes, now you can just call the train/predict functions.

- set the inputs of train/predict functions to the bare minimum. All relevant parameters (learning rate, epochs, batch size, etc.) are displayed in the file "unet_parameters.py". You can change them from there.

- added early stopping

- adjusted inputs of train/predict to be folders instead of list of image files

- possibly other minor modifications that I don't recall right now
