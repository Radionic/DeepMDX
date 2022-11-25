import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import glob
from pathlib import Path

from src import spec_utils
from src.lr_scheduler import NoamLR
from src.nets import CascadedNetWithGAN
from inference import inference

from icecream import ic

class DeepMDX(pl.LightningModule):
    def __init__(self,
                 batch_size=4,
                 num_workers=12,
                 nout=32,
                 nout_lstm=128,
                 n_fft=2048,
                 lr=1e-3,
                 warmup_steps=400,
                 lambda_aux = 1e-2,
                 lambda_adv = 1e-3,
                 use_bn=True,
                 train_discriminator=True,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.use_bn = use_bn
        self.n_fft = n_fft
        self.nout = nout
        self.nout_lstm = nout_lstm
        self.lambda_aux = lambda_aux
        self.lambda_adv = lambda_adv
        self.train_discriminator = train_discriminator
        self.save_hyperparameters()
    
        self.model = CascadedNetWithGAN(self.n_fft, self.nout, self.nout_lstm, use_bn=self.use_bn)
        self.generator_criterion = nn.L1Loss()
        self.discriminator_criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        return self.generator(x)
    
    def configure_optimizers(self):
        optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.lr)
        optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr)
        
        scheduler_D = NoamLR(optimizer_D, warmup_steps=self.warmup_steps)
        scheduler_G = NoamLR(optimizer_G, warmup_steps=self.warmup_steps)
        
        lr_dict_D = {
            "scheduler": scheduler_D,
            "interval": "step",
            "name": "lr_D",
            "frequency": 1,
        }
        lr_dict_G = {
            "scheduler": scheduler_G,
            "interval": "step",
            "name": "lr_G",
            "frequency": 1,
        }
        return [
            {
                "optimizer": optimizer_D,
                "lr_scheduler": lr_dict_D,
            },
            {
                "optimizer": optimizer_G,
                "lr_scheduler": lr_dict_G,
            }
        ]
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        pred, aux = self.model.generator(x)
        
        if optimizer_idx == 0:
            d_loss = 0
            loss_real = 0
            loss_fake = 0
            dis_fake_dist = []
            dis_real_dist = []
            
            if self.train_discriminator:
                pred_real = self.model.discriminator_forward(x, y)
                pred_fake = self.model.discriminator_forward(x, (pred * x).detach())
                
                loss_real = self.discriminator_criterion(pred_real, torch.ones_like(pred_real))
                loss_fake = self.discriminator_criterion(pred_fake, torch.zeros_like(pred_fake))
                d_loss = (loss_real + loss_fake) / 2
            
            
            if (self.global_step // 2) % self.trainer.log_every_n_steps ==0:
                
                if self.train_discriminator:
                    dis_fake_dist = torch.sigmoid(pred_fake).detach().cpu().numpy().flatten().tolist()
                    dis_real_dist = torch.sigmoid(pred_real).detach().cpu().numpy().flatten().tolist()
                wandb.log({
                    "discriminator/dis_real_dist": wandb.Histogram(dis_real_dist),
                    "discriminator/dis_fake_dist": wandb.Histogram(dis_fake_dist),
                }, step=self.global_step)
            
            self.log_dict({
                "discriminator/loss": d_loss,
                "discriminator/dis_real_loss": loss_real,
                "discriminator/dis_fake_loss": loss_fake,
            })
            
            return d_loss if self.train_discriminator else None

        if optimizer_idx == 1:
            content_loss = self.generator_criterion(pred * x, y)
            aux_loss = self.generator_criterion(aux * x, y)
            pred_fake = self.model.discriminator_forward(x, pred * x)
            adv_loss = 0
            if self.train_discriminator:
                adv_loss = self.discriminator_criterion(pred_fake, torch.ones_like(pred_fake))
            g_loss = content_loss + self.lambda_aux * aux_loss + self.lambda_adv * adv_loss
            
            if ((self.global_step - 1) // 2) % self.trainer.log_every_n_steps ==0:
                gen_fake_dist = torch.sigmoid(pred_fake).detach().cpu().numpy().flatten().tolist()
                wandb.log({
                    "generator/gen_dist": wandb.Histogram(gen_fake_dist),
                }, step=self.global_step)
            
            self.log_dict({
                "generator/loss": g_loss,
                "generator/content_loss": content_loss,
                "generator/aux_loss": aux_loss,
                "generator/adv_loss": adv_loss,
            })
            
            return g_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        pred = self.model.generator(x)
        y = spec_utils.crop_center(y, pred)
        loss = self.generator_criterion(pred, y)
        
        self.log("val/loss", loss)
        self.log("val_loss", loss, logger=False)
        
        return loss
    
    def validation_epoch_end(self, outputs):
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
    
    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
        )
        self.log("grad_norm", grad_norm)
    