import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .utils.evaluate import evaluate
from .unet import UNet
from .utils.data_loading import BasicDataset
from .utils.dice_score import dice_loss

from .utils.early_stopping import EarlyStopper
from .unet_parameters import *

def train_model(
        dir_img,
        dir_mask,
        dir_checkpoint,
        model,
        device
):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta) ## early stopping
    global_step = 0

    plt_loss = [] ##

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img', disable = True) as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                        
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        validation_loss = float(1 - val_score)
                        plt_loss.append(validation_loss)
                        print('\r'+' '*128, end = '') # to clear previous outputs
                        print(f'\rValidation loss: {validation_loss}', end = '')

                        # print(f'val dice loss: {validation_loss}')
                        if early_stopper.early_stop(validation_loss):
                            print('\nEARLY STOPPING!')
                            return plt_loss
                        if early_stopper.new_best: # save model with best performance
                            print(' <-- model saved (best performance so far)', end = '')
                            state_dict = model.state_dict()
                            state_dict['mask_values'] = dataset.mask_values
                            torch.save(state_dict, dir_checkpoint)
    print() # just to go down one line

    return plt_loss

def train(dir_img,
          dir_mask,
          dir_checkpoint,
          load = False,
          classes = 2):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)
    model = model.to(memory_format=torch.channels_last)

    if load:
        state_dict = torch.load(load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)

    model.to(device=device)
    return train_model( ## output is training_history
        dir_img = dir_img,
        dir_mask = dir_mask,
        dir_checkpoint = dir_checkpoint,
        model=model,
        device = device
    )


