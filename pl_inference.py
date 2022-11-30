import argparse
import torch
from inference import inference
import pytorch_lightning as pl
from src.pl_deepmdxgan import DeepMDXGAN
import glob
from pathlib import Path
from loguru import logger
from pydub import AudioSegment
from icecream import ic

# model = DeepMDXGAN.load_from_checkpoint('checkpoints/extended_mse/epoch=75-val_loss=0.00044.ckpt')
# if torch.cuda.is_available():
#     model = model.cuda()

# val_paths = glob.glob('./*.wav')
# save_dir = './log/image'
# for val_path in val_paths:
#     audio_name = Path(val_path).stem
#     instrument_wave, vocal_wave, instrument_image, vocal_image = inference(model.model, model.device, val_path, save_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained-model', '-P', type=str, default='checkpoints/best.ckpt')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output_dir', '-o', type=str, default=".")
    args = p.parse_args()
    
    filepath = Path(args.input)
    fileparent = filepath.parent
    filename = filepath.stem
    extension = filepath.suffix
    ic(filepath, fileparent, filename, extension)
    if not args.input.endswith('.wav'):
        logger.info('Converting the audio file to wav...')
        audio = AudioSegment.from_mp3(f"./{str(filepath)}")
        audio.export(f'{fileparent}/{filename}.wav', format='wav')
        filepath = f'{fileparent}/{filename}.wav'
        
    logger.info("Loading model...")
    model = DeepMDXGAN.load_from_checkpoint(args.pretrained_model)
    
    if torch.cuda.is_available():
        logger.info("Using GPU...")
        model = model.cuda()
    
    instrument_wave, vocal_wave, instrument_image, vocal_image = inference(model.model, model.device, filepath, args.output_dir)

    return f"{args.output_dir}/{filename}_Instruments.wav", f"{args.output_dir}/{filename}_Vocals.wav"

if __name__ == '__main__':
    main()