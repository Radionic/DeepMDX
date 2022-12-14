import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from src import dataset
from src import nets
from src import spec_utils
from src import utils


class Separator(object):

    def __init__(self, model, device, batchsize, cropsize, postprocess=False):
        self.model = model
        self.offset = model.generator.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    def _separate(self, X_mag_pad, roi_size):
        X_dataset = []
        patches = (X_mag_pad.shape[2] - 2 * self.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_mag_crop = X_mag_pad[:, :, start:start + self.cropsize]
            X_dataset.append(X_mag_crop)

        X_dataset = np.asarray(X_dataset)

        self.model.eval()
        with torch.no_grad():
            mask = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)

                pred = self.model.generator.predict_mask(X_batch)

                pred = pred.detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)

            mask = np.concatenate(mask, axis=2)

        return mask

    def _preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _postprocess(self, mask, X_mag, X_phase):
        if self.postprocess:
            mask = spec_utils.merge_artifacts(mask)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

        return y_spec, v_spec

    def separate(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)
        mask = mask[:, :, :n_frame]

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

    def separate_tta(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask_tta = self._separate(X_mag_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec


def inference(model, device, input, output_dir, sr=44100, hop_length=1024, n_fft=2048, batchsize=4, cropsize=256, tta=False, postprocess=False, output_image=True):
    print('loading wave source...', end=' ')
    X, sr = librosa.load(input, sr, False, dtype=np.float32, res_type='kaiser_fast')
    basename = os.path.splitext(os.path.basename(input))[0]
    os.makedirs(output_dir, exist_ok=True)

    if X.ndim == 1:
        # mono to stereo
        X = np.asarray([X, X])

    print('stft of wave source...', end=' ')
    X_spec = spec_utils.wave_to_spectrogram(X, hop_length, n_fft)

    sp = Separator(model, device, batchsize, cropsize, postprocess)

    if tta:
        y_spec, v_spec = sp.separate_tta(X_spec)
    else:
        y_spec, v_spec = sp.separate(X_spec)

    print('inverse stft of instruments...', end=' ')
    instrument_wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=hop_length)

    sf.write(f'{output_dir}/{basename}_Instruments.wav', instrument_wave.T, sr)

    print('inverse stft of vocals...', end=' ')
    vocal_wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=hop_length)
    sf.write(f'{output_dir}/{basename}_Vocals.wav', vocal_wave.T, sr)

    if output_image:
        instrument_image = spec_utils.spectrogram_to_image(y_spec)
        utils.imwrite(f'{output_dir}/{basename}_Instruments.jpg', instrument_image)

        vocal_image = spec_utils.spectrogram_to_image(v_spec)
        utils.imwrite(f'{output_dir}/{basename}_Vocals.jpg', vocal_image)
    return instrument_wave, vocal_wave, instrument_image, vocal_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--output_dir', '-o', type=str, default="")
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedNetWithGAN(args.n_fft, 32, 128)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device)["state_dict"], strict=False)
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
    
    inference(
      model,
      device,
      args.input,
      args.sr,
      args.hop_length,
      args.n_fft,
      args.batchsize,
      args.cropsize,
      args.tta,
      args.postprocess,
      args.output_image
    )


if __name__ == '__main__':
    main()
