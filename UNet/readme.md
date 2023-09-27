# U-Net

This is the code for the U-Net! An example of use is shown in "example.ipynb".

The original code is from https://github.com/milesial/Pytorch-UNet [^1].

[^1]: That code is in my opinion inefficient and a little messy. But if my code is not clear you should refer to that since I've built from it.

On top of it, I've added the following modifications:

- Reorganised it so it fits in a single, nice folder called "UNet";

```bash
from UNet.train import train
```

```bash
from UNet.predict import predict
```

- 