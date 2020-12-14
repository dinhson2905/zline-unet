import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from eval import eval_net
from unet import UNet
from losses import DiceBCELoss
from log import setup_logger

g_logger = setup_logger('train_logger', 'log/train.log')
loss_logger = setup_logger('loss_logger', 'log/loss.log')

dir_img_train = 'data/Train/Images/'
dir_mask_train = 'data/Train/Masks/'
dir_img_val = 'data/Val/Images/'
dir_mask_val = 'data/Val/Masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net, device, epochs=5, batch_size=1, lr=0.001, save_cp=True, img_scale=0.5):
    dataset_train = BasicDataset(dir_img_train, dir_mask_train, img_scale)
    dataset_val = BasicDataset(dir_img_val, dir_mask_val, img_scale)

    n_train = len(dataset_train)
    n_val = len(dataset_val)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size,shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    g_logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = DiceBCELoss()

    running_loss = 0.0
    for epoch in range(epochs):
        net.train()
        g_logger.info(f'Epoch {epoch + 1}/{epochs}:')
        loss_logger.info(f'Epoch {epoch + 1}/{epochs}:')
        for param_group in optimizer.param_groups:
            lr_log = param_group['lr']
            g_logger.info(f'learning rate: {lr_log}')
            
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the image are loaded correctly'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                running_loss += loss.item()
                # loss_logger.info(f'{loss.item()}')
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step() # update weight

                pbar.update(imgs.shape[0])
                global_step += 1

                # for 20 minibatch -> statistic
                if global_step % (n_train // (20 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar(
                        'learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        g_logger.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        g_logger.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                    
                    loss_statistic = running_loss / (20 * batch_size)
                    loss_logger.info(f'{loss_statistic}')
                    running_loss = 0.0

        
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                g_logger.info('Created checkpoint directory')
            except OSError:
                pass

            torch.save(net.state_dict(), dir_checkpoint +
                       f'CP2_epoch{epoch + 1}.pth')
            g_logger.info(f'Checkpoint {epoch + 1} saved!')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    return parser.parse_args()


if __name__ == "__main__":
    # logging.basicConfig(filename='train.log', level=logging.INFO,
    #                     format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g_logger.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    g_logger.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        g_logger.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, device=device, img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        g_logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
