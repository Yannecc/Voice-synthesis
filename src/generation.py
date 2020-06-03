import numpy as np
import librosa
import argparse
import torch
import sys
import time


mel = np.loadtxt('mel.csv', delimiter = ',')
print(mel.shape)
sample_rate = 16000
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()


with torch.no_grad():
	sequence = torch.from_numpy(mel).to(device='cuda', dtype=torch.float32)
	sequence = sequence.unsqueeze(0)
	generated_wav = waveglow.infer(sequence)





fpath = "outputs/TEST.wav"
generated_wav = generated_wav.cpu().numpy().astype(np.float32).T
print(generated_wav.shape)
librosa.output.write_wav(fpath, generated_wav, sample_rate,)
