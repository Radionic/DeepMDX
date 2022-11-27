import torch
from torch import nn
import torch.nn.functional as F

from src.utils import _weights_init
from src.layers import Encoder, Decoder, Conv2DBNActiv, LSTMModule, ASPPModule

class BaseNet(nn.Module):

    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6)), use_bn=True):
        super(BaseNet, self).__init__()
        self.enc1 = Conv2DBNActiv(nin, nout, 3, 1, 1, use_bn=use_bn)
        self.enc2 = Encoder(nout, nout * 2, 3, 2, 1, use_bn=use_bn)
        self.enc3 = Encoder(nout * 2, nout * 4, 3, 2, 1, use_bn=use_bn)
        self.enc4 = Encoder(nout * 4, nout * 6, 3, 2, 1, use_bn=use_bn)
        self.enc5 = Encoder(nout * 6, nout * 8, 3, 2, 1, use_bn=use_bn)

        self.aspp = ASPPModule(nout * 8, nout * 8, dilations, dropout=True, use_bn=use_bn)

        self.dec4 = Decoder(nout * (6 + 8), nout * 6, 3, 1, 1, use_bn=use_bn)
        self.dec3 = Decoder(nout * (4 + 6), nout * 4, 3, 1, 1, use_bn=use_bn)
        self.dec2 = Decoder(nout * (2 + 4), nout * 2, 3, 1, 1, use_bn=use_bn)
        self.lstm_dec2 = LSTMModule(nout * 2, nin_lstm, nout_lstm, use_bn=use_bn)
        self.dec1 = Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1, use_bn=use_bn)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = self.aspp(e5)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)

        return h


class CascadedNet(nn.Module):

    def __init__(self, n_fft, nout=32, nout_lstm=128, use_bn=True):
        super(CascadedNet, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.offset = 64
        nin_lstm = self.max_bin // 2

        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, nin_lstm // 2, nout_lstm, use_bn=use_bn),
            Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0, use_bn=use_bn)
        )
        self.stg1_high_band_net = BaseNet(
            2, nout // 4, nin_lstm // 2, nout_lstm // 2, use_bn=use_bn
        )

        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, nin_lstm // 2, nout_lstm, use_bn=use_bn),
            Conv2DBNActiv(nout, nout // 2, 1, 1, 0, use_bn=use_bn)
        )
        self.stg2_high_band_net = BaseNet(
            nout // 4 + 2, nout // 2, nin_lstm // 2, nout_lstm // 2, use_bn=use_bn
        )

        self.stg3_full_band_net = BaseNet(
            3 * nout // 4 + 2, nout, nin_lstm, nout_lstm, use_bn=use_bn
        )

        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

        _weights_init(self.out)
        _weights_init(self.aux_out)

    def forward(self, x):
        # x: (B, 2, H, W)
        # max_bin: 1024

        # (B, 2, max_bin, W), where 2 is stereo
        x = x[:, :, :self.max_bin]

        # ----- Low and High Band Training in Stage 1 -----
        # bandw: 512
        bandw = x.size()[2] // 2
        # (B, 2, bandw, W)
        l1_in = x[:, :, :bandw]
        # (B, 2, bandw, W)
        h1_in = x[:, :, bandw:]
        # (B, 8, bandw, W)
        l1 = self.stg1_low_band_net(l1_in)
        # (B, 8, bandw, W)
        h1 = self.stg1_high_band_net(h1_in)
        # (B, 8, bandw * 2, W)
        aux1 = torch.cat([l1, h1], dim=2)

        # ----- Low and High Band Training in Stage 2 -----
        # (B, 10, bandw, W)
        l2_in = torch.cat([l1_in, l1], dim=1)
        # (B, 10, bandw, W)
        h2_in = torch.cat([h1_in, h1], dim=1)
        # (B, 16, bandw, W)
        l2 = self.stg2_low_band_net(l2_in)
        # (B, 16, bandw, W)
        h2 = self.stg2_high_band_net(h2_in)
        # (B, 16, bandw * 2, W)
        aux2 = torch.cat([l2, h2], dim=2)

        # ----- Full Band Training -----
        # (B, 26, bandw * 2, W)
        f3_in = torch.cat([x, aux1, aux2], dim=1)
        # (B, 32, bandw * 2, W)
        f3 = self.stg3_full_band_net(f3_in)

        # (B, 2, max_bin, W)
        mask = torch.sigmoid(self.out(f3))
        # (B, 2, max_bin + 1, W)
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode='replicate'
        )

        if self.training:
            # (B, 24, max_bin, W)
            aux = torch.cat([aux1, aux2], dim=1)
            # (B, 2, max_bin, W)
            aux = torch.sigmoid(self.aux_out(aux))
            # (B, 2, max_bin + 1, W)
            aux = F.pad(
                input=aux,
                pad=(0, 0, 0, self.output_bin - aux.size()[2]),
                mode='replicate'
            )
            return mask, aux
        else:
            return mask

    def predict_mask(self, x):
        mask = self.forward(x)

        if self.offset > 0:
            mask = mask[:, :, :, self.offset:-self.offset]
            assert mask.size()[3] > 0

        return mask

    def predict(self, x):
        mask = self.forward(x)
        pred_mag = x * mask

        if self.offset > 0:
            pred_mag = pred_mag[:, :, :, self.offset:-self.offset]
            assert pred_mag.size()[3] > 0

        return pred_mag

class CascadedNetWithGAN(nn.Module):
    def __init__(self, n_fft, nout=32, nout_lstm=128, use_bn=True):
        super(CascadedNetWithGAN, self).__init__()
        self.generator = CascadedNet(n_fft, nout, nout_lstm, use_bn=use_bn)
        # self.discriminator = Discriminator(4)
        self.discriminator = Discriminator(2)
      
    def set_requires_grad(self, net, requires_grad):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def generator_forward(self, x):
        return self.generator(x)

    # def discriminator_forward(self, x, y):
    #     # (B, 2, max_bin, W)
    #     x = x[:, :, :self.generator.max_bin]
    #     # (B, 2, max_bin, W)
    #     y = y[:, :, :self.generator.max_bin]
    #     input = torch.cat([x, y], dim=1)
    #     return self.discriminator(input)
    def discriminator_forward(self, y):
        # (B, 2, max_bin, W)
        y = y[:, :, :self.generator.max_bin]
        return self.discriminator(y)

class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        in_filters = in_channels
        layers = []
        for i, out_filters in enumerate([64, 128, 256, 512]):
            if i != 0:
                layers.append(Conv2DBNActiv(in_filters, out_filters, 3, 1, 1, activ=nn.LeakyReLU))
            else:
                layers.append(Conv2DBNActiv(in_filters, out_filters, 3, 1, 1, activ=nn.LeakyReLU, use_bn=False))
            layers.append(Conv2DBNActiv(out_filters, out_filters, 3, 2, 1, activ=nn.LeakyReLU))
            in_filters = out_filters
        layers.append(nn.Conv2d(in_filters, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
