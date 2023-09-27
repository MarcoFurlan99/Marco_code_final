# model parameters
epochs = 5
batch_size = 1
learning_rate = 1e-5
val_percent = 0.1
img_scale = 1.0
amp = True # use mixed precision (GradScaler)
bilinear = False # use bilinear interpolation for upsampling, https://towardsdatascience.com/understanding-u-net-61276b10f360

# early stopping parameters. IMPORTANT: patience counter progressed once every 20% of an epoch
patience = 5
min_delta = 0

# optimizer parameters (RMSprop, with ReduceLROnPlateau scheduler)
weight_decay = 0 # 1e-8
momentum = 0.999
gradient_clipping = 1.0

# predict parameters
out_threshold = .5

# loss function: CrossEntropyLoss + Dice Loss

