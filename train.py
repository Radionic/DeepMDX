import argparse
from datetime import datetime
import json
import logging
import os
import random
import wandb
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from lib import dataset
from lib import nets
from lib import spec_utils
from inference import inference
from tqdm import tqdm
from pathlib import Path


def setup_logger(name, logfile='LOGFILENAME.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(logfile, encoding='utf8')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def train_epoch(dataloader, model, device, optimizers, use_adv_loss=False, train_discriminator=False, train_generator=False):
    model.train()
    optimizer_G, optimizer_D = optimizers
    sum_loss_l1 = 0
    sum_loss_gen_adv = 0
    sum_loss_dis_real = 0
    sum_loss_dis_fake = 0
    pixel_crit = nn.L1Loss()
    gan_crit = nn.BCEWithLogitsLoss()

    gen_dist = []
    dis_real_dist = []
    dis_fake_dist = []

    for itr, (X_batch, y_batch) in enumerate(tqdm(dataloader)):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred, aux = model.generator_forward(X_batch)

        # ----- Train Discriminator -----
        if train_discriminator:
            model.set_requires_grad(model.discriminator, True)
            pred_real = model.discriminator_forward(X_batch, y_batch)
            loss_real = gan_crit(pred_real, torch.ones_like(pred_real))
            dis_real_dist += torch.sigmoid(pred_real).detach().cpu().numpy().flatten().tolist()

            pred_fake = model.discriminator_forward(X_batch, (pred * X_batch).detach())
            loss_fake = gan_crit(pred_fake, torch.zeros_like(pred_fake))
            dis_fake_dist += torch.sigmoid(pred_fake).detach().cpu().numpy().flatten().tolist()

            loss = 0.5 * (loss_real + loss_fake)
            loss.backward()
            optimizer_D.step()
            model.discriminator.zero_grad()

            sum_loss_dis_real += loss_real.item() * len(X_batch)
            sum_loss_dis_fake += loss_fake.item() * len(X_batch)

        # ----- Train Generator ------
        if train_generator:
            loss_main = pixel_crit(pred * X_batch, y_batch)
            loss_aux = pixel_crit(aux * X_batch, y_batch)
            if use_adv_loss:
                model.set_requires_grad(model.discriminator, False)
                pred_fake = model.discriminator_forward(X_batch, pred * X_batch)
                loss_gan = gan_crit(pred_fake, torch.ones_like(pred_fake))
                gen_dist += torch.sigmoid(pred_fake).detach().cpu().numpy().flatten().tolist()
                loss = loss_main * 0.8 + loss_aux * 0.1 + loss_gan * 0.1
            else:
                loss = loss_main * 0.8 + loss_aux * 0.2
            loss.backward()
            optimizer_G.step()
            model.generator.zero_grad()

            # The ratio 8:2 here is for comparsion on previous models without GAN 
            sum_loss_l1 += (loss_main * 0.8 + loss_aux * 0.2).item() * len(X_batch)
            if use_adv_loss:
                sum_loss_gen_adv += loss_gan.item() * len(X_batch)

    dateset_len = len(dataloader.dataset)
    return {
      'train/loss': sum_loss_l1 / dateset_len,
      'train/gen_adv_loss': sum_loss_gen_adv / dateset_len,
      'train/dis_real_loss': sum_loss_dis_real / dateset_len,
      'train/dis_fake_loss': sum_loss_dis_fake / dateset_len,
      'train/gen_dist': wandb.Histogram(gen_dist),
      'train/dis_real_dist':wandb.Histogram(dis_real_dist),
      'train/dis_fake_dist': wandb.Histogram(dis_fake_dist),
    }

def validate_epoch(dataloader, model, device):
    model.eval()
    sum_loss = 0
    pixel_crit = nn.L1Loss()

    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model.generator.predict(X_batch)

            y_batch = spec_utils.crop_center(y_batch, pred)
            loss = pixel_crit(pred, y_batch)

            sum_loss += loss.item() * len(X_batch)

    return {
      'val/loss': sum_loss / len(dataloader.dataset)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--seed', '-s', type=int, default=2019)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--dataset', '-d', required=True)
    p.add_argument('--split_mode', '-S', type=str, choices=['random', 'subdirs'], default='random')
    p.add_argument('--learning_rate', '-l', type=float, default=1e-3)
    p.add_argument('--lr_min', type=float, default=1e-4)
    p.add_argument('--lr_decay_factor', type=float, default=0.9)
    p.add_argument('--lr_decay_patience', type=int, default=6)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--accumulation_steps', '-A', type=int, default=1)
    p.add_argument('--cropsize', '-C', type=int, default=256)
    p.add_argument('--patches', '-p', type=int, default=16)
    p.add_argument('--val_rate', '-v', type=float, default=0.2)
    p.add_argument('--val_filelist', '-V', type=str, default=None)
    p.add_argument('--val_batchsize', '-b', type=int, default=6)
    p.add_argument('--val_cropsize', '-c', type=int, default=256)
    p.add_argument('--num_workers', '-w', type=int, default=48)
    p.add_argument('--epoch', '-E', type=int, default=200)
    p.add_argument('--reduction_rate', '-R', type=float, default=0.0)
    p.add_argument('--reduction_level', '-L', type=float, default=0.2)
    p.add_argument('--mixup_rate', '-M', type=float, default=0.0)
    p.add_argument('--mixup_alpha', '-a', type=float, default=1.0)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    logger.debug(vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    val_filelist = []
    if args.val_filelist is not None:
        with open(args.val_filelist, 'r', encoding='utf8') as f:
            val_filelist = json.load(f)

    train_filelist, val_filelist = dataset.train_val_split(
        dataset_dir=args.dataset,
        split_mode=args.split_mode,
        val_rate=args.val_rate,
        val_filelist=val_filelist
    )

    if args.debug:
        logger.info('### DEBUG MODE')
        train_filelist = train_filelist[:1]
        val_filelist = val_filelist[:1]
    elif args.val_filelist is None and args.split_mode == 'random':
        with open('val_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(val_filelist, f, ensure_ascii=False)

    for i, (X_fname, y_fname) in enumerate(val_filelist):
        logger.info('{} {} {}'.format(i + 1, os.path.basename(X_fname), os.path.basename(y_fname)))

    device = torch.device('cpu')
    model = nets.CascadedNetWithGAN(args.n_fft)
    if args.pretrained_model is not None:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    optimizer_G = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.generator.parameters()),
        lr=args.learning_rate
    )
    optimizer_D = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.discriminator.parameters()),
        lr=args.learning_rate
    )
    optimizers = [optimizer_G, optimizer_D]

    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G,
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        threshold=1e-6,
        min_lr=args.lr_min,
        verbose=True
    )
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D,
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        threshold=1e-6,
        min_lr=args.lr_min,
        verbose=True
    )

    bins = args.n_fft // 2 + 1
    freq_to_bin = 2 * bins / args.sr
    unstable_bins = int(200 * freq_to_bin)
    stable_bins = int(22050 * freq_to_bin)
    reduction_weight = np.concatenate([
        np.linspace(0, 1, unstable_bins, dtype=np.float32)[:, None],
        np.linspace(1, 0, stable_bins - unstable_bins, dtype=np.float32)[:, None],
        np.zeros((bins - stable_bins, 1), dtype=np.float32),
    ], axis=0) * args.reduction_level

    training_set = dataset.make_training_set(
        filelist=train_filelist,
        sr=args.sr,
        hop_length=args.hop_length,
        n_fft=args.n_fft
    )

    train_dataset = dataset.VocalRemoverTrainingSet(
        training_set * args.patches,
        cropsize=args.cropsize,
        reduction_rate=args.reduction_rate,
        reduction_weight=reduction_weight,
        mixup_rate=args.mixup_rate,
        mixup_alpha=args.mixup_alpha
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers
    )

    patch_list = dataset.make_validation_set(
        filelist=val_filelist,
        cropsize=args.val_cropsize,
        sr=args.sr,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        offset=model.generator.offset
    )

    val_dataset = dataset.VocalRemoverValidationSet(
        patch_list=patch_list
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers
    )

    log = []
    best_loss = np.inf
    for epoch in range(args.epoch):
        logger.info('# epoch {}'.format(epoch))
        
        use_adv_loss = epoch >= 15
        train_discriminator = True # epoch >= 10
        train_generator = epoch >= 10
        train_loss = train_epoch(train_dataloader, model, device, optimizers, use_adv_loss, train_discriminator, train_generator)
        val_loss = validate_epoch(val_dataloader, model, device)

        log_data = {
          **train_loss,
          **val_loss
        }
        wandb.log(log_data)
        if epoch % 5 == 0 or epoch == args.epoch - 1:
            log_data = {}
            val_paths = glob.glob('/project/asc2022/plus/DeepMDX/data/val/*.wav')
            save_dir = '/project/asc2022/plus/DeepMDX'
            for val_path in val_paths:
                audio_name = Path(val_path).stem
                instrument_wave, vocal_wave, instrument_image, vocal_image = inference(model, device, val_path, save_dir)
                log_data[f'val/instrument_wav_{audio_name}'] = wandb.Audio(instrument_wave.T, sample_rate=44100)
                log_data[f'val/vocal_wav_{audio_name}'] = wandb.Audio(vocal_wave.T, sample_rate=44100)
                log_data[f'val/instrument_spectrogram_{audio_name}'] = wandb.Image(instrument_image)
                log_data[f'val/vocal_spectrogram_{audio_name}'] = wandb.Image(vocal_image)
            wandb.log(log_data)

        train_loss = train_loss['train/loss']
        val_loss = val_loss['val/loss']
        if train_generator:
            scheduler_G.step(val_loss)
        if train_discriminator:
            scheduler_D.step(val_loss)
        logger.info(f'  * training loss = {train_loss:.6f}, validation loss = {val_loss:.6f}')

        model_path = f'models/model_iter{epoch}.pth'
        if epoch % 5 == 0 or epoch == args.epoch - 1:
            torch.save(model.state_dict(), model_path)
        if val_loss < best_loss:
            best_loss = val_loss
            logger.info('  * best validation loss')

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))
    wandb.init(project="Music Demixing")

    try:
        main()
    except Exception as e:
        logger.exception(e)
