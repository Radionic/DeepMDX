import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import glob
from pathlib import Path

from src import spec_utils
from src.lr_scheduler import NoamLR
from src.nets import CascadedNetNoGAN
from inference import inference

from icecream import ic

class DeepMDX(pl.LightningModule):
    def __init__(self,
                 batch_size=4,
                 num_workers=12,
                 nout=32,
                 nout_lstm=128,
                 n_fft=2048,
                 lr=1e-4,
                 lr_min=1e-5,
                 lr_check_interval=1000,
                 lr_decay_factor=0.9,
                 lr_decay_patience=2,
                 warmup_steps_g=400, # For NoamLR
                 lambda_aux = 2e-1,
                 use_bn=True,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.lr_min = lr_min
        self.lr_check_interval = lr_check_interval
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_patience = lr_decay_patience
        self.warmup_steps_g = warmup_steps_g
        self.use_bn = use_bn
        self.n_fft = n_fft
        self.nout = nout
        self.nout_lstm = nout_lstm
        self.lambda_aux = lambda_aux
        self.save_hyperparameters()
    
        self.model = CascadedNetNoGAN(self.n_fft, self.nout, self.nout_lstm, use_bn=self.use_bn)
        self.generator_criterion = nn.L1Loss()
        
    
    def forward(self, x):
        return self.generator(x)
    
    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr)
        
        # scheduler_G = NoamLR(optimizer_G, warmup_steps=self.warmup_steps_g)
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_G,
            factor=self.lr_decay_factor,
            patience=self.lr_decay_patience,
            threshold=1e-6,
            min_lr=self.lr_min,
            verbose=True
        )
    
        lr_dict_G = {
            "scheduler": scheduler_G,
            "interval": "epoch",
            "monitor": "val_loss",
            "name": "lr_G",
            "frequency": self.lr_check_interval,
        }
        return [
            {
                "optimizer": optimizer_G,
                "lr_scheduler": lr_dict_G,
            }
        ]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        pred, aux = self.model.generator(x)

        content_loss = self.generator_criterion(pred * x, y)
        aux_loss = self.generator_criterion(aux * x, y)
        
        g_loss = content_loss + self.lambda_aux * aux_loss
        g_loss = g_loss * self.g_loss_scale
        
        self.log_dict({
            "generator/loss": g_loss,
            "generator/content_loss": content_loss,
            "generator/aux_loss": aux_loss,
        })
        
        return g_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        pred = self.model.generator.predict(x)
        y = spec_utils.crop_center(y, pred)
        loss = self.generator_criterion(pred, y)
        
        self.log("val/loss", loss)
        self.log("val_loss", loss, logger=False)
        
        return loss
    
    def on_train_epoch_end(self):
        if self.current_epoch % 5 == 0:
            log_data = {}
            val_paths = glob.glob('./data/val/*.wav')
            save_dir = './log/image'
            for val_path in val_paths:
                audio_name = Path(val_path).stem
                instrument_wave, vocal_wave, instrument_image, vocal_image = inference(self.model, self.device, val_path, save_dir)
                log_data[f'val_wav/instrument_wav_{audio_name}'] = wandb.Audio(instrument_wave.T, sample_rate=44100)
                log_data[f'val_wav/vocal_wav_{audio_name}'] = wandb.Audio(vocal_wave.T, sample_rate=44100)
                log_data[f'val_spec/instrument_spectrogram_{audio_name}'] = wandb.Image(instrument_image)
                log_data[f'val_spec/vocal_spectrogram_{audio_name}'] = wandb.Image(vocal_image)
            wandb.log(log_data)        
            self.model.train()
    
    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
        )
        self.log("grad_norm", grad_norm)
    