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
                 lr_g=1e-3,
                 lr_d = 1e-4,
                 warmup_steps_g=400,
                 warmup_steps_d=400,
                 lambda_aux = 2e-1,
                 lambda_adv = 1e-3,
                 use_bn=True,
                 train_discriminator=True,
                 discriminator_step=1,
                 generator_step=1,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.warmup_steps_g = warmup_steps_g
        self.warmup_steps_d = warmup_steps_d
        self.use_bn = use_bn
        self.n_fft = n_fft
        self.nout = nout
        self.nout_lstm = nout_lstm
        self.lambda_aux = lambda_aux
        self.lambda_adv = lambda_adv
        self.train_discriminator = train_discriminator
        self.discriminator_step = discriminator_step
        self.generator_step = generator_step
        self.save_hyperparameters()
    
        self.model = CascadedNetWithGAN(self.n_fft, self.nout, self.nout_lstm, use_bn=self.use_bn)
        self.generator_criterion = nn.L1Loss()
        self.discriminator_criterion = nn.BCEWithLogitsLoss()
        
        self.gen_fake_dist = []
        self.dis_real_dist = []
        self.dis_fake_dist = []
        
    
    def forward(self, x):
        return self.generator(x)
    
    def configure_optimizers(self):
        optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.lr_d)
        optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr_g)
        
        # scheduler_D = NoamLR(optimizer_D, warmup_steps=self.warmup_steps_d)
        # scheduler_G = NoamLR(optimizer_G, warmup_steps=self.warmup_steps_g)
        scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_D,
            factor=0.9,
            patience=3,
            threshold=1e-6,
            min_lr=1e-5,
            verbose=True
        )
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_G,
            factor=0.9,
            patience=3,
            threshold=1e-6,
            min_lr=1e-5,
            verbose=True
        )
        
        lr_dict_D = {
            "scheduler": scheduler_D,
            "interval": "epoch",
            "monitor": "val_loss",
            "name": "lr_D",
            "frequency": 2,
        }
        lr_dict_G = {
            "scheduler": scheduler_G,
            "interval": "epoch",
            "monitor": "val_loss",
            "name": "lr_G",
            "frequency": 2,
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
        
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if optimizer_idx == 0:
            for _ in range(self.discriminator_step):
                optimizer.step(closure=optimizer_closure)

        if optimizer_idx == 1:
            for _ in range(self.generator_step):
                optimizer.step(closure=optimizer_closure)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        pred, aux = self.model.generator(x)
        
        if optimizer_idx == 0:
            d_loss = 0
            loss_real = 0
            loss_fake = 0
            
            if self.train_discriminator:
                # pred_real = self.model.discriminator_forward(x, y)
                # pred_fake = self.model.discriminator_forward(x, (pred * x).detach())
                pred_real = self.model.discriminator_forward(y)
                pred_fake = self.model.discriminator_forward((pred * x).detach())
                
                # Gan Loss
                loss_real = self.discriminator_criterion(pred_real, torch.ones_like(pred_real))
                loss_fake = self.discriminator_criterion(pred_fake, torch.zeros_like(pred_fake))
                
                # Relativistic Gan Loss
                # loss_real = self.discriminator_criterion(pred_real - pred_fake.mean(0), torch.ones_like(pred_real))
                # loss_fake = self.discriminator_criterion(pred_fake - pred_real.mean(0), torch.zeros_like(pred_fake))
                d_loss = (loss_real + loss_fake) / 2
                
                self.dis_fake_dist += torch.sigmoid(pred_fake).detach().cpu().numpy().flatten().tolist()
                self.dis_real_dist += torch.sigmoid(pred_real).detach().cpu().numpy().flatten().tolist()
            
            
            if (self.global_step // 2) % self.trainer.log_every_n_steps ==0:                    
                wandb.log({
                    "discriminator/dis_real_dist": wandb.Histogram(self.dis_real_dist),
                    "discriminator/dis_fake_dist": wandb.Histogram(self.dis_fake_dist),
                }, step=self.global_step)
                self.dis_fake_dist.clear()
                self.dis_real_dist.clear()
            
            self.log_dict({
                "discriminator/loss": d_loss,
                "discriminator/dis_real_loss": loss_real,
                "discriminator/dis_fake_loss": loss_fake,
            })
            
            return d_loss if self.train_discriminator else None

        if optimizer_idx == 1:
            content_loss = self.generator_criterion(pred * x, y)
            aux_loss = self.generator_criterion(aux * x, y)
            # pred_fake = self.model.discriminator_forward(x, pred * x)
            pred_fake = self.model.discriminator_forward(pred * x)
            adv_loss = 0
            if self.train_discriminator:
                adv_loss = self.discriminator_criterion(pred_fake, torch.ones_like(pred_fake))
            g_loss = content_loss + self.lambda_aux * aux_loss + self.lambda_adv * adv_loss
            
            self.gen_fake_dist += torch.sigmoid(pred_fake).detach().cpu().numpy().flatten().tolist()
            if ((self.global_step - 1) // 2) % self.trainer.log_every_n_steps ==0:
                wandb.log({
                    "generator/gen_dist": wandb.Histogram(self.gen_fake_dist),
                }, step=self.global_step)
                self.gen_fake_dist.clear()
            
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
    