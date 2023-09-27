
from UNet.unet.unet_model import UNet
import torch

def BN_adapt(model_root, dataset, device, saving_root):
    model = UNet(n_channels=3, n_classes=2).to(device=device)
    state_dict = torch.load(model_root, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)

    model.train()
    with torch.no_grad():
        for batch in dataset :
            images = batch['image'].to(device=device, dtype=torch.float32)
            _ = model(images)
    torch.save(model.state_dict(), saving_root)
